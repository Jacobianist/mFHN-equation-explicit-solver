#pragma once

// #include <chrono>
#include <fstream>
// #include <iomanip>
// #include <iostream>
#include <sstream>
#include <string>

class Logger
{
    std::ofstream logFile;

   public:
    Logger(const std::string& filename)
    {
        logFile.open(filename);
        log("=== Simulation Started ===");
    }
    ~Logger()
    {
        log("=== Simulation Finished ===");
        logFile.close();
    }
    // Шаблон для вывода любого сообщения
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
    void logMemoryUsage(int N)
    {
        // 6 массивов на Device (u, v, w, u_next, v_next, w_next)
        // 3 массива на Host (h_u, h_v, h_w) для вывода
        const int numArrays = 9;
        const int bytesPerValue = sizeof(float);  // 4 байта
        double totalMB = (double)N * numArrays * bytesPerValue / (1024.0 * 1024.0);
        std::stringstream ss;
        ss << "Estimated Memory Usage: " << std::fixed << std::setprecision(2) << totalMB << " MB (" << numArrays << " arrays of size " << N << ")";
        log(ss.str());
    }
};
