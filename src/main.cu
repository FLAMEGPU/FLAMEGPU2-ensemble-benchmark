#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>

#include "flamegpu/flame_api.h"

/**
 * FLAME GPU 2 implementation of the Boids model, using spatial3D messaging.
 * This is based on the FLAME GPU 1 implementation, but with dynamic generation of agents. 
 * Agents are also clamped to be within the environment bounds, rather than wrapped as in FLAME GPU 1.
 * 
 * @todo - Should the agent's velocity change when it is clamped to the environment?
 */


/**
 * Get the length of a vector
 * @param x x component of the vector
 * @param y y component of the vector
 * @param z z component of the vector
 * @return the length of the vector
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
    return sqrtf(x * x + y * y + z * z);
}

/**
 * Add a scalar to a vector in-place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param z z component of the vector
 * @param value scalar value to add
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Add(float &x, float &y, float &z, const float value) {
    x += value;
    y += value;
    z += value;
}

/**
 * Subtract a scalar from a vector in-place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param z z component of the vector
 * @param value scalar value to subtract
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Sub(float &x, float &y, float &z, const float value) {
    x -= value;
    y -= value;
    z -= value;
}

/**
 * Multiply a vector by a scalar value in-place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param z z component of the vector
 * @param multiplier scalar value to multiply by
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Mult(float &x, float &y, float &z, const float multiplier) {
    x *= multiplier;
    y *= multiplier;
    z *= multiplier;
}

/**
 * Divide a vector by a scalar value in-place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param z z component of the vector
 * @param divisor scalar value to divide by
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
    x /= divisor;
    y /= divisor;
    z /= divisor;
}

/**
 * Normalize a 3 component vector in-place
 * @param x x component of the vector
 * @param y y component of the vector
 * @param z z component of the vector
 */ 
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Normalize(float &x, float &y, float &z) {
    // Get the length
    float length = vec3Length(x, y, z);
    vec3Div(x, y, z, length);
}

/**
 * Clamp each component of a 3-part position to lie within a minimum and maximum value.
 * Performs the operation in place
 * Unlike the FLAME GPU 1 example, this is a clamping operation, rather than wrapping.
 * @param x x component of the vector
 * @param y y component of the vector
 * @param z z component of the vector
 * @param MIN_POSITION the minimum value for each component
 * @param MAX_POSITION the maximum value for each component
 */
FLAMEGPU_HOST_DEVICE_FUNCTION void clampPosition(float &x, float &y, float &z, const float MIN_POSITION, const float MAX_POSITION) {
    x = (x < MIN_POSITION)? MIN_POSITION: x;
    x = (x > MAX_POSITION)? MAX_POSITION: x;

    y = (y < MIN_POSITION)? MIN_POSITION: y;
    y = (y > MAX_POSITION)? MAX_POSITION: y;

    z = (z < MIN_POSITION)? MIN_POSITION: z;
    z = (z > MAX_POSITION)? MAX_POSITION: z;
}

FLAMEGPU_INIT_FUNCTION(Init) {
    auto env = FLAMEGPU->environment;
    std::mt19937 rngEngine(1234);
    std::uniform_real_distribution<float> position_distribution(env.getProperty<float>("MIN_POSITION"), env.getProperty<float>("MAX_POSITION"));
    std::uniform_real_distribution<float> velocity_distribution(-1, 1);
    std::uniform_real_distribution<float> velocity_magnitude_distribution(env.getProperty<float>("MIN_INITIAL_SPEED"), env.getProperty<float>("MAX_INITIAL_SPEED"));
   
    const unsigned int POPULATION_TO_GENERATE = FLAMEGPU->environment.getProperty<unsigned int>("POPULATION_TO_GENERATE");
    for (unsigned int i = 0; i < POPULATION_TO_GENERATE; ++i) {
        auto agent = FLAMEGPU->newAgent("Boid");
        agent.setVariable<int>("id", i);

        agent.setVariable<float>("x", position_distribution(rngEngine));
        agent.setVariable<float>("y", position_distribution(rngEngine));
        agent.setVariable<float>("z", position_distribution(rngEngine));

        // Generate a random velocity direction
        float fx = velocity_distribution(rngEngine);
        float fy = velocity_distribution(rngEngine);
        float fz = velocity_distribution(rngEngine);
        // Generate a random speed between 0 and the maximum initial speed
        float fmagnitude = velocity_magnitude_distribution(rngEngine);
        // Use the random speed for the velocity.
        vec3Normalize(fx, fy, fz);
        vec3Mult(fx, fy, fz, fmagnitude);

        // Set these for the agent.
        agent.setVariable<float>("fx", fx);
        agent.setVariable<float>("fy", fy);
        agent.setVariable<float>("fz", fz);
    }
}

/**
 * outputdata agent function for Boid agents, which outputs publicly visible properties to a message list
 */
const char* outputdata = R"###(
FLAMEGPU_AGENT_FUNCTION(outputdata, MsgNone, MsgSpatial3D) {
    // Output each agents publicly visible properties.
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
    FLAMEGPU->message_out.setVariable<float>("fz", FLAMEGPU->getVariable<float>("fz"));
    return ALIVE;
}
)###";
/**
 * inputdata agent function for Boid agents, which reads data from neighbouring Boid agents, to perform the boid flocking model.
 */
const char* inputdata = R"###(
// Vector utility functions, see top of file for versions with commentary
FLAMEGPU_HOST_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
    return sqrtf(x * x + y * y + z * z);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Add(float &x, float &y, float &z, const float value) {
    x += value;
    y += value;
    z += value;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Sub(float &x, float &y, float &z, const float value) {
    x -= value;
    y -= value;
    z -= value;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Mult(float &x, float &y, float &z, const float multiplier) {
    x *= multiplier;
    y *= multiplier;
    z *= multiplier;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
    x /= divisor;
    y /= divisor;
    z /= divisor;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Normalize(float &x, float &y, float &z) {
    // Get the length
    float length = vec3Length(x, y, z);
    vec3Div(x, y, z, length);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void clampPosition(float &x, float &y, float &z, const float MIN_POSITION, const float MAX_POSITION) {
    x = (x < MIN_POSITION)? MIN_POSITION: x;
    x = (x > MAX_POSITION)? MAX_POSITION: x;

    y = (y < MIN_POSITION)? MIN_POSITION: y;
    y = (y > MAX_POSITION)? MAX_POSITION: y;

    z = (z < MIN_POSITION)? MIN_POSITION: z;
    z = (z > MAX_POSITION)? MAX_POSITION: z;
}
// Agent function
FLAMEGPU_AGENT_FUNCTION(inputdata, MsgSpatial3D, MsgNone) {

    // Agent properties in local register
    int id = FLAMEGPU->getVariable<int>("id");
    // Agent position
    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    float agent_z = FLAMEGPU->getVariable<float>("z");
    // Agent velocity
    float agent_fx = FLAMEGPU->getVariable<float>("fx");
    float agent_fy = FLAMEGPU->getVariable<float>("fy");
    float agent_fz = FLAMEGPU->getVariable<float>("fz");

    // Boids percieved center
    float perceived_centre_x = 0.0f;
    float perceived_centre_y = 0.0f;
    float perceived_centre_z = 0.0f;
    int perceived_count = 0;

    // Boids global velocity matching
    float global_velocity_x = 0.0f;
    float global_velocity_y = 0.0f;
    float global_velocity_z = 0.0f;

    // Boids short range avoidance centre
    float collision_centre_x = 0.0f;
    float collision_centre_y = 0.0f;
    float collision_centre_z = 0.0f;
    int collision_count = 0;

    const float INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("INTERACTION_RADIUS");
    const float SEPARATION_RADIUS = FLAMEGPU->environment.getProperty<float>("SEPARATION_RADIUS");
    // Iterate location messages, accumulating relevant data and counts.
    for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
        // Ignore self messages.
        if (message.getVariable<int>("id") != id) {
            // Get the message location and velocity.
            const float message_x = message.getVariable<float>("x");
            const float message_y = message.getVariable<float>("y");
            const float message_z = message.getVariable<float>("z");
            const float message_fx = message.getVariable<float>("fx");
            const float message_fy = message.getVariable<float>("fy");
            const float message_fz = message.getVariable<float>("fz");

            // Check interaction radius
            float separation = vec3Length(agent_x - message_x, agent_y - message_y, agent_z - message_z);

            if (separation < (INTERACTION_RADIUS)) {
                // Update the percieved centre
                perceived_centre_x += message_x;
                perceived_centre_y += message_y;
                perceived_centre_z += message_z;
                perceived_count++;

                // Update percieved velocity matching
                global_velocity_x += message_fx;
                global_velocity_y += message_fy;
                global_velocity_z += message_fz;

                // Update collision centre
                if (separation < (SEPARATION_RADIUS)) {  // dependant on model size
                    collision_centre_x += message_x;
                    collision_centre_y += message_y;
                    collision_centre_z += message_z;
                    collision_count += 1;
                }
            }
        }
    }

    // Divide positions/velocities by relevant counts.
    vec3Div(perceived_centre_x, perceived_centre_y, perceived_centre_z, perceived_count);
    vec3Div(global_velocity_x, global_velocity_y, global_velocity_z, perceived_count);
    vec3Div(global_velocity_x, global_velocity_y, global_velocity_z, collision_count);

    // Total change in velocity
    float velocity_change_x = 0.f;
    float velocity_change_y = 0.f;
    float velocity_change_z = 0.f;

    // Rule 1) Steer towards perceived centre of flock (Cohesion)
    float steer_velocity_x = 0.f;
    float steer_velocity_y = 0.f;
    float steer_velocity_z = 0.f;
    if (perceived_count > 0) {
        const float STEER_SCALE = FLAMEGPU->environment.getProperty<float>("STEER_SCALE");
        steer_velocity_x = (perceived_centre_x - agent_x) * STEER_SCALE;
        steer_velocity_y = (perceived_centre_y - agent_y) * STEER_SCALE;
        steer_velocity_z = (perceived_centre_z - agent_z) * STEER_SCALE;
    }
    velocity_change_x += steer_velocity_x;
    velocity_change_y += steer_velocity_y;
    velocity_change_z += steer_velocity_z;

    // Rule 2) Match neighbours speeds (Alignment)
    float match_velocity_x = 0.f;
    float match_velocity_y = 0.f;
    float match_velocity_z = 0.f;
    if (collision_count > 0) {
        const float MATCH_SCALE = FLAMEGPU->environment.getProperty<float>("MATCH_SCALE");
        match_velocity_x = global_velocity_x * MATCH_SCALE;
        match_velocity_y = global_velocity_y * MATCH_SCALE;
        match_velocity_z = global_velocity_z * MATCH_SCALE;
    }
    velocity_change_x += match_velocity_x;
    velocity_change_y += match_velocity_y;
    velocity_change_z += match_velocity_z;

    // Rule 3) Avoid close range neighbours (Separation)
    float avoid_velocity_x = 0.0f;
    float avoid_velocity_y = 0.0f;
    float avoid_velocity_z = 0.0f;
    if (collision_count > 0) {
        const float COLLISION_SCALE = FLAMEGPU->environment.getProperty<float>("COLLISION_SCALE");
        avoid_velocity_x = (agent_x - collision_centre_x) * COLLISION_SCALE;
        avoid_velocity_y = (agent_y - collision_centre_y) * COLLISION_SCALE;
        avoid_velocity_z = (agent_z - collision_centre_z) * COLLISION_SCALE;
    }
    velocity_change_x += avoid_velocity_x;
    velocity_change_y += avoid_velocity_y;
    velocity_change_z += avoid_velocity_z;

    // Global scale of velocity change
    vec3Mult(velocity_change_x, velocity_change_y, velocity_change_z, FLAMEGPU->environment.getProperty<float>("GLOBAL_SCALE"));

    // Update agent velocity
    agent_fx += velocity_change_x;
    agent_fy += velocity_change_y;
    agent_fz += velocity_change_z;

    // Bound velocity
    float agent_fscale = vec3Length(agent_fx, agent_fy, agent_fz);
    if (agent_fscale > 1) {
        vec3Div(agent_fx, agent_fy, agent_fz, agent_fscale);
    }

    // Apply the velocity
    const float TIME_SCALE = FLAMEGPU->environment.getProperty<float>("TIME_SCALE");
    agent_x += agent_fx * TIME_SCALE;
    agent_y += agent_fy * TIME_SCALE;
    agent_z += agent_fz * TIME_SCALE;

    // Bound position
    clampPosition(agent_x, agent_y, agent_z, FLAMEGPU->environment.getProperty<float>("MIN_POSITION"), FLAMEGPU->environment.getProperty<float>("MAX_POSITION"));

    // Update global agent memory.
    FLAMEGPU->setVariable<float>("x", agent_x);
    FLAMEGPU->setVariable<float>("y", agent_y);
    FLAMEGPU->setVariable<float>("z", agent_z);

    FLAMEGPU->setVariable<float>("fx", agent_fx);
    FLAMEGPU->setVariable<float>("fy", agent_fy);
    FLAMEGPU->setVariable<float>("fz", agent_fz);

    return ALIVE;
}
)###";

int main(int argc, const char ** argv) {
    
    unsigned int initialPopSize = 500;
    unsigned int finalPopSize = 1000;
    unsigned int popSizeIncrement = 250;
    
    unsigned int totalRuns = 40;
    std::vector<unsigned int> ensembleSizes = {1, 4, 10, 20, 40};

    std::vector<double> serialResults;
    std::vector<double> concurrentResults;
    
    for (unsigned int popSize = initialPopSize; popSize <= finalPopSize; popSize += popSizeIncrement) {
        for (unsigned int ensembleSize : ensembleSizes) {
            std::cout << "Staring run with popSize: " << popSize << ", species: " << ensembleSize << std::endl;
            ModelDescription model("Boids_Ensemble");

            /**
            * GLOBALS
            */
            EnvironmentDescription &env = model.Environment();
            {
                // Population size to generate, if no agents are loaded from disk
                env.newProperty("POPULATION_TO_GENERATE", popSize);

                // Environment Bounds
                env.newProperty("MIN_POSITION", -0.5f);
                env.newProperty("MAX_POSITION", +0.5f);

                // Initialisation parameter(s)
                env.newProperty("MAX_INITIAL_SPEED", 1.0f);
                env.newProperty("MIN_INITIAL_SPEED", 0.01f);

                // Interaction radius
                env.newProperty("INTERACTION_RADIUS", 0.1f);
                env.newProperty("SEPARATION_RADIUS", 0.005f);

                // Global Scalers
                env.newProperty("TIME_SCALE", 0.0005f);
                env.newProperty("GLOBAL_SCALE", 0.15f);

                // Rule scalers
                env.newProperty("STEER_SCALE", 0.65f);
                env.newProperty("COLLISION_SCALE", 0.75f);
                env.newProperty("MATCH_SCALE", 1.25f);
            }

            {   // Location message      
                std::string messageName = "location";
                MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>(messageName);
                // Set the range and bounds.
                message.setRadius(env.getProperty<float>("INTERACTION_RADIUS"));
                message.setMin(env.getProperty<float>("MIN_POSITION"), env.getProperty<float>("MIN_POSITION"), env.getProperty<float>("MIN_POSITION"));
                message.setMax(env.getProperty<float>("MAX_POSITION"), env.getProperty<float>("MAX_POSITION"), env.getProperty<float>("MAX_POSITION"));
                // A message to hold the location of an agent.
                message.newVariable<int>("id");
                // X Y Z are implicit.
                // message.newVariable<float>("x");
                // message.newVariable<float>("y");
                // message.newVariable<float>("z");
                message.newVariable<float>("fx");
                message.newVariable<float>("fy");
                message.newVariable<float>("fz");
            }
            {   // Boid agent
                std::string agentName("Boid");
                AgentDescription &agent = model.newAgent(agentName);
                agent.newVariable<int>("id");
                agent.newVariable<float>("x");
                agent.newVariable<float>("y");
                agent.newVariable<float>("z");
                agent.newVariable<float>("fx");
                agent.newVariable<float>("fy");
                agent.newVariable<float>("fz");
                std::string messageName = "location";
                std::string outputFuncName = "outputdata";
                std::string inputFuncName = "inputdata";
                agent.newRTCFunction(agentName + outputFuncName, outputdata).setMessageOutput(messageName);
                agent.newRTCFunction(agentName + inputFuncName, inputdata).setMessageInput(messageName);
            }

            /**
            * Control flow
            */     
            model.addInitFunction(Init);
            {   // Layer #1
                LayerDescription &layer = model.newLayer();
                std::string agentName = "Boid";
                std::string outputFuncName = "outputdata";
                layer.addAgentFunction(agentName, agentName + outputFuncName);
            }
            {   // Layer #2
                LayerDescription &layer = model.newLayer();
                std::string agentName = "Boid";
                std::string inputFuncName = "inputdata";
                layer.addAgentFunction(agentName, agentName + inputFuncName);
            }

            /**
            * Execution
            */
            std::cout << "Running ensemble size: " << ensembleSize << std::endl;
            unsigned int runsRemaining = totalRuns;
            const auto startTime = std::chrono::system_clock::now();
            while (runsRemaining) {
                RunPlanVec runs(model, ensembleSize);
                {
                    runs.setSteps(500);
                }
                
                CUDAEnsemble cuda_ensemble(model, argc, argv);              
                cuda_ensemble.simulate(runs);
                runsRemaining -= ensembleSize;
            }
            const auto endTime = std::chrono::system_clock::now();
            const auto runTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            
            std::cout << "Run complete. Total run time: " << runTime << "ms" << std::endl;
            serialResults.push_back(runTime);

            /**
            * Export Pop
            */
            // cuda_model.exportData("end.xml");

#ifdef VISUALISATION
            visualisation.join();
            visualisation.close();
#endif
        }
    }

    // Output parameters to file
    std::ofstream paramsFile("params.csv");
    if (paramsFile.is_open()) {
        paramsFile << initialPopSize << "," << finalPopSize << "," << popSizeIncrement << std::endl;
    }

    // Output concurrent results to file
    std::ofstream concurrentOutputFile("concurrent.csv");
    if (concurrentOutputFile.is_open()) {
        unsigned int i = 0;
        for (unsigned int popSize = initialPopSize; popSize <= finalPopSize; popSize += popSizeIncrement) {
            for (unsigned int ensembleSize : ensembleSizes) {
                std::string separator = ensembleSize == ensembleSizes.back() ? "\n" : ", ";
                concurrentOutputFile << concurrentResults[i] << separator;
                i++;
            }
        }
    }

    // Output serial results to file
    std::ofstream serialOutputFile("serial.csv");
    if (serialOutputFile.is_open()) {
        unsigned int i = 0;
        for (unsigned int popSize = initialPopSize; popSize <= finalPopSize; popSize += popSizeIncrement) {
            for (unsigned int ensembleSize : ensembleSizes) {
                std::string separator = ensembleSize == ensembleSizes.back() ? "\n" : ", ";
                serialOutputFile << serialResults[i] << separator;
                i++;
            }
        }
    }
    return 0;
}


