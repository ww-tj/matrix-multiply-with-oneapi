#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <vector>

// 定义矩阵的大小
const int matrix_size = 100;

// 生成随机矩阵
std::vector<std::vector<double>> generateRandomMatrix(int rows, int cols)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

// 计算矩阵乘法
std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>> &matrix1, const std::vector<std::vector<double>> &matrix2)
{
    int rows1 = matrix1.size();
    int cols1 = matrix1[0].size();
    int cols2 = matrix2[0].size();

    std::vector<std::vector<double>> result(rows1, std::vector<double>(cols2));

    // 使用 SYCL 编程模型
    {
        // 创建 SYCL 的队列和设备选择器
        sycl::queue queue(sycl::gpu_selector_v);

        // 创建输入和输出缓冲区

        cl::sycl::buffer<double, 2> buffer1(matrix1.data()->data(), cl::sycl::range<2>(rows1, cols1));
        cl::sycl::buffer<double, 2> buffer2(matrix2.data()->data(), cl::sycl::range<2>(cols1, cols2));
        cl::sycl::buffer<double, 2> buffer_result(result.data()->data(), cl::sycl::range<2>(rows1, cols2));

        // 提交 SYCL 的计算任务
        queue.submit([&](cl::sycl::handler &cgh) {
            auto accessor1 = buffer1.get_access<cl::sycl::access::mode::read>(cgh);
            auto accessor2 = buffer2.get_access<cl::sycl::access::mode::read>(cgh);
            auto accessor_result = buffer_result.get_access<cl::sycl::access::mode::write>(cgh);

            cgh.parallel_for<class MatrixMultiplyKernel>(cl::sycl::range<2>(rows1, cols2), [=](cl::sycl::item<2> item) {
                int i = item[0];
                int j = item[1];
                double sum = 0.0;
                for (int k = 0; k < cols1; ++k) {
                    sum += accessor1[i][k] * accessor2[k][j];
                }
                accessor_result[i][j] = sum;
            });
        });

        // 同步等待计算任务完成
        queue.wait_and_throw();
    }

    return result;
}

int main()
{
    // 生成随机矩阵
    std::vector<std::vector<double>> matrix1 = generateRandomMatrix(matrix_size, matrix_size);
    std::vector<std::vector<double>> matrix2 = generateRandomMatrix(matrix_size, matrix_size);

    // 进行1000次随机矩阵乘法并计时
    auto start_time = std::chrono::high_resolution_clock::now();
    int total_iterations = 1000;
    int progress_interval = total_iterations / 100; // 计算进度的显示间隔
    for (int i = 0; i < total_iterations; ++i) {
        std::vector<std::vector<double>> result = matrixMultiply(matrix1, matrix2);
        if ((i + 1) % progress_interval == 0) {
            // 清屏后显示
            std::cout << "\033[2J\033[1;1H";
            double progress = (static_cast<double>(i + 1) / total_iterations) * 100;
            std::cout << "进度: " << progress << "%" << std::endl;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    // 计算总时间（以秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_time = duration.count() / 1000.0;

    // 打印结果
    std::cout << "总时间: " << total_time << "秒" << std::endl;

    return 0;
}
