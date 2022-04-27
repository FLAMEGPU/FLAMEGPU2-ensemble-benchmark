#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>

#include "flamegpu/flamegpu.h"

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
    std::mt19937_64 rngEngine(FLAMEGPU->random.getSeed());
    std::uniform_real_distribution<float> position_distribution(env.getProperty<float>("MIN_POSITION"), env.getProperty<float>("MAX_POSITION"));
    std::uniform_real_distribution<float> velocity_distribution(-1, 1);
    std::uniform_real_distribution<float> velocity_magnitude_distribution(env.getProperty<float>("MIN_INITIAL_SPEED"), env.getProperty<float>("MAX_INITIAL_SPEED"));
   
    const unsigned int POPULATION_TO_GENERATE = FLAMEGPU->environment.getProperty<unsigned int>("POPULATION_TO_GENERATE");
    auto agent_type = FLAMEGPU->agent("Boid");
    for (unsigned int i = 0; i < POPULATION_TO_GENERATE; ++i) {
        auto agent = agent_type.newAgent();
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
FLAMEGPU_AGENT_FUNCTION(outputdata, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    // Output each agents publicly visible properties.
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
    FLAMEGPU->message_out.setVariable<float>("fz", FLAMEGPU->getVariable<float>("fz"));
    return flamegpu::ALIVE;
}
)###";
const char* outputdataBruteForce = R"###(
FLAMEGPU_AGENT_FUNCTION(outputdata, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    // Output each agents publicly visible properties.
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
    FLAMEGPU->message_out.setVariable<float>("fz", FLAMEGPU->getVariable<float>("fz"));
    return flamegpu::ALIVE;
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
FLAMEGPU_AGENT_FUNCTION(inputdata, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    // Agent properties in local register
    const flamegpu::id_t id = FLAMEGPU->getID();
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

    // Total change in velocity
    float velocity_change_x = 0.f;
    float velocity_change_y = 0.f;
    float velocity_change_z = 0.f;

    const float INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("INTERACTION_RADIUS");
    const float SEPARATION_RADIUS = FLAMEGPU->environment.getProperty<float>("SEPARATION_RADIUS");
    // Iterate location messages, accumulating relevant data and counts.
    for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
        // Ignore self messages.
        if (message.getVariable<flamegpu::id_t>("id") != id) {
            // Get the message location and velocity.
            const float message_x = message.getVariable<float>("x");
            const float message_y = message.getVariable<float>("y");
            const float message_z = message.getVariable<float>("z");

            // Check interaction radius
            float separation = vec3Length(agent_x - message_x, agent_y - message_y, agent_z - message_z);

            if (separation < INTERACTION_RADIUS) {
                // Update the percieved centre
                perceived_centre_x += message_x;
                perceived_centre_y += message_y;
                perceived_centre_z += message_z;
                perceived_count++;

                // Update percieved velocity matching
                const float message_fx = message.getVariable<float>("fx");
                const float message_fy = message.getVariable<float>("fy");
                const float message_fz = message.getVariable<float>("fz");
                global_velocity_x += message_fx;
                global_velocity_y += message_fy;
                global_velocity_z += message_fz;

                // Update collision centre
                if (separation < (SEPARATION_RADIUS)) {  // dependant on model size
                    // Rule 3) Avoid other nearby boids (Separation)
                    float normalizedSeparation = (separation / SEPARATION_RADIUS);
                    float invNormSep = (1.0f - normalizedSeparation);
                    float invSqSep = invNormSep * invNormSep;

                    const float collisionScale = FLAMEGPU->environment.getProperty<float>("COLLISION_SCALE");
                    velocity_change_x += collisionScale * (agent_x - message_x) * invSqSep;
                    velocity_change_y += collisionScale * (agent_y - message_y) * invSqSep;
                    velocity_change_z += collisionScale * (agent_z - message_z) * invSqSep;
                }
            }
        }
    }

    if (perceived_count) {
        // Divide positions/velocities by relevant counts.
        vec3Div(perceived_centre_x, perceived_centre_y, perceived_centre_z, perceived_count);
        vec3Div(global_velocity_x, global_velocity_y, global_velocity_z, perceived_count);

        // Rule 1) Steer towards perceived centre of flock (Cohesion)
        float steer_velocity_x = 0.f;
        float steer_velocity_y = 0.f;
        float steer_velocity_z = 0.f;

        const float STEER_SCALE = FLAMEGPU->environment.getProperty<float>("STEER_SCALE");
        steer_velocity_x = (perceived_centre_x - agent_x) * STEER_SCALE;
        steer_velocity_y = (perceived_centre_y - agent_y) * STEER_SCALE;
        steer_velocity_z = (perceived_centre_z - agent_z) * STEER_SCALE;

        velocity_change_x += steer_velocity_x;
        velocity_change_y += steer_velocity_y;
        velocity_change_z += steer_velocity_z;

        // Rule 2) Match neighbours speeds (Alignment)
        float match_velocity_x = 0.f;
        float match_velocity_y = 0.f;
        float match_velocity_z = 0.f;

        const float MATCH_SCALE = FLAMEGPU->environment.getProperty<float>("MATCH_SCALE");
        match_velocity_x = global_velocity_x;
        match_velocity_y = global_velocity_y;
        match_velocity_z = global_velocity_z;

        velocity_change_x += (match_velocity_x - agent_fx) * MATCH_SCALE;
        velocity_change_y += (match_velocity_y - agent_fy) * MATCH_SCALE;
        velocity_change_z += (match_velocity_z - agent_fz) * MATCH_SCALE;
    }

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

    float minSpeed = 0.5f;
    if (agent_fscale < minSpeed) {
        // Normalise
        vec3Div(agent_fx, agent_fy, agent_fz, agent_fscale);

        // Scale to min
        vec3Mult(agent_fx, agent_fy, agent_fz, minSpeed);
    }

    // Steer away from walls - Computed post normalization to ensure good avoidance. Prevents constant term getting swamped
    const float wallInteractionDistance = 0.10f;
    const float wallSteerStrength = 0.05f;
    const float minPosition = FLAMEGPU->environment.getProperty<float>("MIN_POSITION");
    const float maxPosition = FLAMEGPU->environment.getProperty<float>("MAX_POSITION");

    if (agent_x - minPosition < wallInteractionDistance) {
        agent_fx += wallSteerStrength;
    }
    if (agent_y - minPosition < wallInteractionDistance) {
        agent_fy += wallSteerStrength;
    }
    if (agent_z - minPosition < wallInteractionDistance) {
        agent_fz += wallSteerStrength;
    }

    if (maxPosition - agent_x < wallInteractionDistance) {
        agent_fx -= wallSteerStrength;
    }
    if (maxPosition - agent_y < wallInteractionDistance) {
        agent_fy -= wallSteerStrength;
    }
    if (maxPosition - agent_z < wallInteractionDistance) {
        agent_fz -= wallSteerStrength;
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

    return flamegpu::ALIVE;
}
)###";
const char* inputdataBruteForce = R"###(
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
FLAMEGPU_AGENT_FUNCTION(inputdata, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    // Agent properties in local register
    const flamegpu::id_t id = FLAMEGPU->getID();
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

    // Total change in velocity
    float velocity_change_x = 0.f;
    float velocity_change_y = 0.f;
    float velocity_change_z = 0.f;

    const float INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("INTERACTION_RADIUS");
    const float SEPARATION_RADIUS = FLAMEGPU->environment.getProperty<float>("SEPARATION_RADIUS");
    // Iterate location messages, accumulating relevant data and counts.
    for (const auto &message : FLAMEGPU->message_in) {
        // Ignore self messages.
        if (message.getVariable<flamegpu::id_t>("id") != id) {
            // Get the message location and velocity.
            const float message_x = message.getVariable<float>("x");
            const float message_y = message.getVariable<float>("y");
            const float message_z = message.getVariable<float>("z");

            // Check interaction radius
            float separation = vec3Length(agent_x - message_x, agent_y - message_y, agent_z - message_z);

            if (separation < INTERACTION_RADIUS) {
                // Update the percieved centre
                perceived_centre_x += message_x;
                perceived_centre_y += message_y;
                perceived_centre_z += message_z;
                perceived_count++;

                // Update percieved velocity matching
                const float message_fx = message.getVariable<float>("fx");
                const float message_fy = message.getVariable<float>("fy");
                const float message_fz = message.getVariable<float>("fz");
                global_velocity_x += message_fx;
                global_velocity_y += message_fy;
                global_velocity_z += message_fz;

                // Update collision centre
                if (separation < (SEPARATION_RADIUS)) {  // dependant on model size
                    // Rule 3) Avoid other nearby boids (Separation)
                    float normalizedSeparation = (separation / SEPARATION_RADIUS);
                    float invNormSep = (1.0f - normalizedSeparation);
                    float invSqSep = invNormSep * invNormSep;

                    const float collisionScale = FLAMEGPU->environment.getProperty<float>("COLLISION_SCALE");
                    velocity_change_x += collisionScale * (agent_x - message_x) * invSqSep;
                    velocity_change_y += collisionScale * (agent_y - message_y) * invSqSep;
                    velocity_change_z += collisionScale * (agent_z - message_z) * invSqSep;
                }
            }
        }
    }

    if (perceived_count) {
        // Divide positions/velocities by relevant counts.
        vec3Div(perceived_centre_x, perceived_centre_y, perceived_centre_z, perceived_count);
        vec3Div(global_velocity_x, global_velocity_y, global_velocity_z, perceived_count);

        // Rule 1) Steer towards perceived centre of flock (Cohesion)
        float steer_velocity_x = 0.f;
        float steer_velocity_y = 0.f;
        float steer_velocity_z = 0.f;

        const float STEER_SCALE = FLAMEGPU->environment.getProperty<float>("STEER_SCALE");
        steer_velocity_x = (perceived_centre_x - agent_x) * STEER_SCALE;
        steer_velocity_y = (perceived_centre_y - agent_y) * STEER_SCALE;
        steer_velocity_z = (perceived_centre_z - agent_z) * STEER_SCALE;

        velocity_change_x += steer_velocity_x;
        velocity_change_y += steer_velocity_y;
        velocity_change_z += steer_velocity_z;

        // Rule 2) Match neighbours speeds (Alignment)
        float match_velocity_x = 0.f;
        float match_velocity_y = 0.f;
        float match_velocity_z = 0.f;

        const float MATCH_SCALE = FLAMEGPU->environment.getProperty<float>("MATCH_SCALE");
        match_velocity_x = global_velocity_x;
        match_velocity_y = global_velocity_y;
        match_velocity_z = global_velocity_z;

        velocity_change_x += (match_velocity_x - agent_fx) * MATCH_SCALE;
        velocity_change_y += (match_velocity_y - agent_fy) * MATCH_SCALE;
        velocity_change_z += (match_velocity_z - agent_fz) * MATCH_SCALE;
    }

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

    float minSpeed = 0.5f;
    if (agent_fscale < minSpeed) {
        // Normalise
        vec3Div(agent_fx, agent_fy, agent_fz, agent_fscale);

        // Scale to min
        vec3Mult(agent_fx, agent_fy, agent_fz, minSpeed);
    }

    // Steer away from walls - Computed post normalization to ensure good avoidance. Prevents constant term getting swamped
    const float wallInteractionDistance = 0.10f;
    const float wallSteerStrength = 0.05f;
    const float minPosition = FLAMEGPU->environment.getProperty<float>("MIN_POSITION");
    const float maxPosition = FLAMEGPU->environment.getProperty<float>("MAX_POSITION");

    if (agent_x - minPosition < wallInteractionDistance) {
        agent_fx += wallSteerStrength;
    }
    if (agent_y - minPosition < wallInteractionDistance) {
        agent_fy += wallSteerStrength;
    }
    if (agent_z - minPosition < wallInteractionDistance) {
        agent_fz += wallSteerStrength;
    }

    if (maxPosition - agent_x < wallInteractionDistance) {
        agent_fx -= wallSteerStrength;
    }
    if (maxPosition - agent_y < wallInteractionDistance) {
        agent_fy -= wallSteerStrength;
    }
    if (maxPosition - agent_z < wallInteractionDistance) {
        agent_fz -= wallSteerStrength;
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

    return flamegpu::ALIVE;
}
)###";

typedef struct Experiment {
    Experiment(std::string title,
        unsigned int initialPopSize, unsigned int finalPopSize, unsigned int popSizeIncrement,
	unsigned int totalRuns,
	std::vector<unsigned int> ensembleSizes,
    int repetitions,
	unsigned int steps,
	bool spatial) {
            this->title = title;
	    this->initialPopSize = initialPopSize;
	    this->finalPopSize = finalPopSize;
	    this->popSizeIncrement = popSizeIncrement;
	    this->totalRuns = totalRuns;
	    this->ensembleSizes = ensembleSizes;
        this->repetitions = repetitions;
	    this->steps = steps;
	    this->spatial = spatial;
	}
	std::string title;
	unsigned int initialPopSize, finalPopSize, popSizeIncrement;
	unsigned int totalRuns;
	std::vector<unsigned int> ensembleSizes;
    int repetitions;
	unsigned int steps;
	bool spatial;
} Experiment;

int main(int argc, const char ** argv) {
    
    const int repetitions = 3;

    Experiment smallPopBruteForce("small_pop_brute_force", 128, 1024, 128, 60, std::vector<unsigned int> {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 20}, repetitions, 500, false);
    Experiment largePopBruteForce("large_pop_brute_force", 2048, 8192, 2048, 60, std::vector<unsigned int> {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 20}, repetitions, 500, false);
    Experiment veryLargePopBruteForce("very_large_pop_brute_force", 5000, 5000, 5000, 60, std::vector<unsigned int> {1, 2, 4, 5, 6, 10, 12, 15, 20, 30, 60}, repetitions, 500, false);
    
    Experiment smallPop("small_pop", 128, 1024, 128, 60, std::vector<unsigned int> {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 20}, repetitions, 500, true);
    Experiment largePop("large_pop", 2048, 8192, 2048, 60, std::vector<unsigned int> {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 20}, repetitions, 500, true);
    Experiment veryLargePop("very_large_pop_brute_force", 5000, 20000, 5000, 60, std::vector<unsigned int> {1, 2, 4, 5, 6, 10, 12, 15, 20, 30, 60}, repetitions, 500, true);
   
    //std::vector<Experiment> experiments = { smallPopBruteForce, largePopBruteForce };
    std::vector<Experiment> experiments = {smallPop, largePop, smallPopBruteForce, largePopBruteForce };
    //std::vector<Experiment> experiments = {smallPop, largePop};

    for (auto experiment : experiments) {

        // Pandas
        std::string csvFileName = experiment.title + ".csv";
        std::ofstream csv(csvFileName);
        csv << "repetition,pop_size,ensemble_size,s_sim_mean" << std::endl;
   
        for (int repetition = -1; repetition < experiment.repetitions; repetition++) {
            std::cout << "Beginning repetiton " << repetition << std::endl;
            for (unsigned int popSize = experiment.initialPopSize; popSize <= experiment.finalPopSize; popSize += experiment.popSizeIncrement) {
                for (unsigned int ensembleSize : experiment.ensembleSizes) {
                    std::cout << "Staring run with popSize: " << popSize << ", concurrent_runs: " << ensembleSize << std::endl;
                    flamegpu::ModelDescription model("Boids_Ensemble");

                    /**
                    * GLOBALS
                    */
                    flamegpu::EnvironmentDescription &env = model.Environment();
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
	        	        if (experiment.spatial) {
                            flamegpu::MessageSpatial3D::Description &message = model.newMessage<flamegpu::MessageSpatial3D>(messageName);
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
                        } else {
	        	            flamegpu::MessageBruteForce::Description &message = model.newMessage<flamegpu::MessageBruteForce>(messageName);
                            // A message to hold the location of an agent.
                            message.newVariable<int>("id");
                            message.newVariable<float>("x");
                            message.newVariable<float>("y");
                            message.newVariable<float>("z");
                            message.newVariable<float>("fx");
                            message.newVariable<float>("fy");
                            message.newVariable<float>("fz");
	        	        }
                    }
                    {   // Boid agent
                        std::string agentName("Boid");
                        flamegpu::AgentDescription &agent = model.newAgent(agentName);
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
	        	        if (experiment.spatial) {
                            agent.newRTCFunction(agentName + outputFuncName, outputdata).setMessageOutput(messageName);
                            agent.newRTCFunction(agentName + inputFuncName, inputdata).setMessageInput(messageName);
	        	        } else {
                            agent.newRTCFunction(agentName + outputFuncName, outputdataBruteForce).setMessageOutput(messageName);
                            agent.newRTCFunction(agentName + inputFuncName, inputdataBruteForce).setMessageInput(messageName);
	        	        }
	        
                    }

                    /**
                    * Control flow
                    */     
                    model.addInitFunction(Init);
                    {   // Layer #1
                        flamegpu::LayerDescription &layer = model.newLayer();
                        std::string agentName = "Boid";
                        std::string outputFuncName = "outputdata";
                        layer.addAgentFunction(agentName, agentName + outputFuncName);
                    }
                    {   // Layer #2
                        flamegpu::LayerDescription &layer = model.newLayer();
                        std::string agentName = "Boid";
                        std::string inputFuncName = "inputdata";
                        layer.addAgentFunction(agentName, agentName + inputFuncName);
                    }

                    /**
                    * Execution
                    */
                    std::cout << "Running ensemble size: " << ensembleSize << std::endl;
                    const auto startTime = std::chrono::system_clock::now();
                    {
                        flamegpu::RunPlanVector runs(model, experiment.totalRuns);
                        {
                            runs.setSteps(experiment.steps);
                            
                            // On the dummy run to force RTC compilation, only need a single step as results are discarded
                            if (repetition < 0) {
                                runs.setSteps(1);
                            }
                            // Set the seeds for the run plan vector. Use the current repetition as the seed for all simulations in the ensemble.
                            runs.setRandomSimulationSeed(repetition, 0);
                        }
                        
                        flamegpu::CUDAEnsemble cuda_ensemble(model, argc, argv);              
                        cuda_ensemble.Config().out_format = "";
                        cuda_ensemble.Config().quiet = true;
                        cuda_ensemble.Config().concurrent_runs = ensembleSize;
                        cuda_ensemble.Config().devices = {0};
                        cuda_ensemble.simulate(runs);
                    }
                    const auto endTime = std::chrono::system_clock::now();
                    const auto runTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
                    
                    std::cout << "Run complete. Total run time: " << runTime / 1000.0 << "s" << std::endl;
                    // Only log if this isn't the dummy RTC compilation run
                    if (repetition >= 0) {
                        csv << repetition << "," << popSize << "," << ensembleSize << "," << runTime / 1000.0 << std::endl;
                    }

#ifdef VISUALISATION
                    visualisation.join();
                    visualisation.close();
#endif
                }
            }
        }  
    }

    return 0;
}
