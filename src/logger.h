#pragma once

// #include <chrono>
#include <fstream>
// #include <iomanip>
// #include <iostream>
#include <sstream>
#include <string>

/**
 * @brief Simple file and console logger for simulation output.
 * 
 * Writes timestamped log messages to both a file and standard output.
 * Automatically logs simulation start/end markers and can estimate memory usage.
 */
class Logger
{
    std::ofstream logFile;  ///< Output file stream for log file

   public:
    /**
     * @brief Constructs a logger and opens the log file.
     * @param filename Path to the log file
     */
    Logger(const std::string& filename)
    {
        logFile.open(filename);
        log("=== Simulation Started ===");
    }

    /**
     * @brief Destructor that logs simulation end and closes the file.
     */
    ~Logger()
    {
        log("=== Simulation Finished ===");
        logFile.close();
    }

    /**
     * @brief Logs a message with timestamp to both file and console.
     * @param message The message to log
     */
    void log(const std::string& message)
    {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        std::string output = "[" + ss.str() + "] " + message;
        std::cout << output << std::endl;
        logFile << output << std::endl;
    }

    /**
     * @brief Logs estimated memory usage for the simulation arrays.
     * 
     * Calculates memory for:
     * - 6 device arrays: u, v, w, u_next, v_next, w_next
     * - 3 host arrays: h_u, h_v, h_w (for output)
     * 
     * @param N Grid size per dimension
     */
    void logMemoryUsage(int N)
    {
        // 6 arrays on Device (u, v, w, u_next, v_next, w_next)
        // 3 arrays on Host (h_u, h_v, h_w) for output
        const int numArrays = 9;
        const int bytesPerValue = sizeof(float);  // 4 bytes
        double totalMB = (double)N * numArrays * bytesPerValue / (1024.0 * 1024.0);
        std::stringstream ss;
        ss << "Estimated Memory Usage: " << std::fixed << std::setprecision(2) << totalMB << " MB (" << numArrays << " arrays of size " << N << ")";
        log(ss.str());
    }
};
