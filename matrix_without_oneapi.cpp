#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// �������Ĵ�С
const int matrix_size = 100;

// �����������
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

// �������˷�
std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>> &matrix1, const std::vector<std::vector<double>> &matrix2)
{
    int rows1 = matrix1.size();
    int cols1 = matrix1[0].size();
    int cols2 = matrix2[0].size();

    std::vector<std::vector<double>> result(rows1, std::vector<double>(cols2));

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cols1; ++k) {
                sum += matrix1[i][k] * matrix2[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

int main()
{
    // �����������
    std::vector<std::vector<double>> matrix1 = generateRandomMatrix(matrix_size, matrix_size);
    std::vector<std::vector<double>> matrix2 = generateRandomMatrix(matrix_size, matrix_size);

    // ����1000���������˷�����ʱ
    auto start_time = std::chrono::high_resolution_clock::now();
    int total_iterations = 1000;
    int progress_interval = total_iterations / 100; // ������ȵ���ʾ���
    for (int i = 0; i < total_iterations; ++i) {
        std::vector<std::vector<double>> result = matrixMultiply(matrix1, matrix2);
        if ((i + 1) % progress_interval == 0) {
            // ��������ʾ
            std::cout << "\033[2J\033[1;1H";
            double progress = (static_cast<double>(i + 1) / total_iterations) * 100;
            std::cout << "����: " << progress << "%" << std::endl;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    // ������ʱ�䣨����Ϊ��λ��
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_time = duration.count() / 1000.0;

    // ��ӡ���
    std::cout << "��ʱ��: " << total_time << "��" << std::endl;

    return 0;
}
