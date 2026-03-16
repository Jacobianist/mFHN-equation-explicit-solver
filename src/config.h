#pragma once
#include <string>

#include "params.h"

/**
 * @brief Configuration loader for simulation parameters.
 *
 * Loads and validates simulation parameters from a JSON configuration file.
 * Performs stability checks and ensures all parameters are within valid ranges.
 */
class ConfigLoader
{
   public:
    /**
     * @brief Loads simulation parameters from a JSON file.
     * @param filename Path to the JSON configuration file
     * @return SimParams structure with loaded parameters
     * @throws std::runtime_error if file cannot be opened or contains invalid data
     */
    static SimParams load(const std::string& filename);

    /**
     * @brief Validates simulation parameters for consistency and stability.
     *
     * Checks:
     * - Dimension is 1 or 2
     * - Grid size N >= 3
     * - dx and dt are positive
     * - steps and num_snapshots are positive
     * - Diffusion coefficients are non-negative
     * - CFL stability condition is satisfied (with warning if violated)
     *
     * @param params Parameters to validate
     * @throws std::runtime_error if validation fails
     */
    static void validate(const SimParams& params);
};
