#include <string>				// string 库
#include <vector>				// vector 容器库
#include <cmath>				// 数学 库
#include <opencv2/highgui.hpp>	// opencv 高级 GUI 库
#include <opencv2/imgproc.hpp>	// opencv 图像处理过程库
using namespace std;			// 标准命名空间
using namespace cv;				// opencv 命名空间

#define PI 3.1415926

// 两旋转矩形中心点距离
double dist(RotatedRect a, RotatedRect b)
{
	double dx = a.center.x - b.center.x;
	double dy = a.center.y - b.center.y;
	return sqrtf(dx * dx + dy * dy);
}

// 两点距离
double dist(Point2f a, Point2f b)
{
	double dx = a.x - b.x;
	double dy = a.y - b.y;
	return sqrtf(dx * dx + dy * dy);
}

// 主函数
int main(int argc, char* argv[])
{
	// -------------------------------------------------------------------------------------
	// ----------↓↓↓↓↓↓↓↓↓---------- 预处理  ----------↓↓↓↓↓↓↓↓↓----------
	// -------------------------------------------------------------------------------------

	// 二维码图片路径
	string srcPath;
	if (argc == 1) srcPath = "pic/9.jpg";
	else srcPath = argv[1];
	
	// 读图 原图存为 src
	Mat src = imread(srcPath);
	if (src.empty()) return 0;				// 判断图像是否为空
	imshow("src", src);
	
	// 转换为灰度图 存为过程图 img
	Mat img = src.clone();					// 原图深拷贝至过程图
	cvtColor(img, img, COLOR_BGR2GRAY);		// 转灰度图
	imshow("gray", img);
	
	// 中值滤波 去椒盐噪声 滤波模板尺寸(5 * 5)
	medianBlur(img, img, 5);
	imshow("medianBlur", img);				

	// 二值化
	string winName = "threshold | press 'q' to continue";	// 窗口名
	int value = 100;										// 二值化阈值
	Mat tmp = Mat::zeros(src.size(), img.type());			// tmp 图像用于绘制过程图临时存储
	namedWindow(winName, 0);								// 创建窗口
	resizeWindow(winName, Size(tmp.cols, tmp.rows + 45));
	createTrackbar("value", winName, &value, 256);			// 创建滚动条
	while (waitKey(33) != 'q')								// 每 33 毫秒检测一次按键 按 'q' 键跳出循环
	{
		threshold(img, tmp, value - 1, 255, THRESH_BINARY);	// 二值化
		imshow(winName, tmp);
	}
	img = tmp.clone();										// 复制 tmp 到 img

	// tmp 变为3通道 BGR 图像 并且宽度变为 2 倍 用于同窗口左右两图对比
	tmp = Mat::zeros(Size(src.cols << 1, img.rows), CV_8UC3);				
	Mat rightPart = tmp(Rect(img.cols, 0, img.cols, img.rows));				// tmp 右半部分
	Mat leftPart = tmp(Rect(0, 0, img.cols, img.rows));						// tmp 左半部分


	// -------------------------------------------------------------------------------------
	// ----------↓↓↓↓↓↓↓↓↓---------- 定位点  ----------↓↓↓↓↓↓↓↓↓----------
	// -------------------------------------------------------------------------------------

	// 边缘检测（查找轮廓）
	vector<vector<Point>> contours;											// 记录轮廓
	vector<Point2f> keypoint;												// 记录定位点中心坐标
	vector<Vec4i> hierarchy;												// 记录轮廓层级关系
	findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);	// opencv 库函数 查找轮廓
	img = src.clone();														// 复制一份原图 用于做半透明定位点绘制

	// 在 tmp 右半部分绘制所有轮廓 白色表示
	for (int i = 0; i < contours.size(); ++i)
		drawContours(rightPart, contours, i, Scalar(255, 255, 255));		

	// 在 tmp 右半部分重新绘制定位点轮廓 绿色表示
	for (int i = 0; i < contours.size(); ++i)
	{
		RotatedRect rect = minAreaRect(Mat(contours[i]));					// 该轮廓的最小包围矩形
		double width = rect.size.width;										// 矩形其中一边	
		double height = rect.size.height;									// 矩形另外一边

		// 判断宽高比
		if (min(width, height) / max(width, height) > 0.8)
		{
			// 判断 子轮廓有且只有一个
			int child_1 = hierarchy[i][2];														
			if (child_1 != -1 && hierarchy[child_1][0] == -1 && hierarchy[child_1][1] == -1)	
			{
				// 判断 孙子轮廓有且只有一个
				int child_2 = hierarchy[child_1][2];											
				if (child_2 != -1 && hierarchy[child_2][0] == -1 && hierarchy[child_2][1] == -1)
				{
					// 外围轮廓、子轮廓、孙子轮廓的最小包围矩形
					RotatedRect rect_child_1 = minAreaRect(Mat(contours[child_1]));
					RotatedRect rect_child_2 = minAreaRect(Mat(contours[child_2]));

					// 三个中心点的距离
					double dist_1 = dist(rect, rect_child_1);
					double dist_2 = dist(rect_child_1, rect_child_2);
					double dist_3 = dist(rect_child_2, rect);

					// 中心点坐标允许的误差范围
					int offset = 5;	

					// 判断 外围轮廓、子轮廓、孙子轮廓的中心在一个点
					if (abs(dist_1 - dist_2) < offset && abs(dist_2 - dist_3) < offset &&abs(dist_3 - dist_1) < offset)
					{
						// 妈祖四个条件后才能判定为二维码的定位点
						// 在 img 上绘制定位点
						drawContours(img, contours, i, Scalar(0, 255, 0), -1);
						// 在 tmp 右半部分绘制定位点
						drawContours(rightPart, contours, i, Scalar(0, 255, 0));
						drawContours(rightPart, contours, child_1, Scalar(0, 255, 0));
						drawContours(rightPart, contours, child_2, Scalar(0, 255, 0));
						// 记录定位点的中心坐标
						keypoint.push_back(Point2f(rect.center.x, rect.center.y));
					}
				}
			}
		}
	}
	
	line(tmp, Point(src.cols, 0), Point(src.cols, src.rows - 1), Scalar(0, 0, 255), 3);	// 在 tmp 中间画红线 区分左右两部分
	addWeighted(src, 0.5, img, 0.5, 0, img);	// 一份绘制过定位点的 和 一份原图 按 0.5 比例混合 达到半透明效果
	img.copyTo(leftPart);						// copy 到左半部分
	imshow("keypoint", tmp);

	// 释放内存
	vector<vector<Point>>().swap(contours);
	vector<Vec4i>().swap(hierarchy);


	// -------------------------------------------------------------------------------------
	// ----------↓↓↓↓↓↓↓↓↓---------- 点配对  ----------↓↓↓↓↓↓↓↓↓----------
	// -------------------------------------------------------------------------------------

	// 如果一幅图像中有 N 个二维码则会检测到 N * 3 个中心点 则需要配对这些中心点
	// 考虑有些情况会检测到 N * 3 ± M 个中心点 
	// 例如 某二维码某定位点未检测到
	// 或者 某二维码被物体遮挡 只露出一两个定位点

	int len_k = 85;									// 模阈值
	int angle_low = 75;								// cos 下限
	int angle_high = 105;							// cos 上限
	winName = "match keypoint";
	namedWindow(winName, 0);
	resizeWindow(winName, Size(img.cols, img.rows + 165));
	createTrackbar("len_k", winName, &len_k, 100);
	createTrackbar("angle_low", winName, &angle_low, 180);
	createTrackbar("angle_high", winName, &angle_high, 180);

	// 记录中心点两辆间距
	vector<vector<double>> d;
	for (int i = 0; i + 1 < keypoint.size(); i++)
	{
		d.push_back(vector<double>(keypoint.size()));
		for (int j = i + 1; j < keypoint.size(); j++)
			d[i][j] = (dist(keypoint[i], keypoint[j]));
	}

	// 按 'q' 结束
	while (waitKey(33) != 'q')
	{
		vector<Point2f> keypointDst;					// 按顺序记录配对点
		vector<int> contoursMask(keypoint.size(), 0);	// 标记是否配对

		// 开始配对
		for (int i = 0; i + 2 < keypoint.size(); i++)
		{
			if (contoursMask[i]) continue;				// 跳过已配对的点
			for (int j = i + 1; j + 1 < keypoint.size(); j++)
			{
				if (contoursMask[j]) continue;			// 跳过已配对的点
				for (int k = j + 1; k < keypoint.size(); k++)
				{
					if (contoursMask[k]) continue;		// 跳过已配对的点

					// 边长
					double a = d[i][j], b = d[i][k], c = d[j][k];

					// 排序
					if (a > b) swap(a, b); 
					if (a > c) swap(a, c); 
					if (b > c) swap(b, c);

					// 角度 余弦定理
					double A = acos((b * b + c * c - a * a) / (2 * b * c)) / PI * 180;
					double B = acos((a * a + c * c - b * b) / (2 * a * c)) / PI * 180;
					double C = acos((a * a + b * b - c * c) / (2 * a * b)) / PI * 180;

					// 判断是否配对 某两边长之比在阈值内 某角在阈值内
					if ((a / b > len_k * 0.01 && angle_low < C && C < angle_high)
						|| (b / c > len_k * 0.01 && angle_low < A && A < angle_high)
						|| (a / c > len_k * 0.01 && angle_low < B && B < angle_high))
					{
						// 标记
						contoursMask[i] = 1; contoursMask[j] = 1; contoursMask[k] = 1;

						// 记录
						keypointDst.push_back(keypoint[i]);
						keypointDst.push_back(keypoint[j]);
						keypointDst.push_back(keypoint[k]);
					}
				}
			}
		}

		// 配对定位点连线 红色表示
		tmp = img.clone();
		for (int i = 0; i + 2 < keypointDst.size(); i += 3)
		{
			Scalar color = Scalar(0, 0, 255);
			line(tmp, keypointDst[i], keypoint[i + 1], color, 2);
			line(tmp, keypointDst[i + 1], keypoint[i + 2], color, 2);
			line(tmp, keypointDst[i + 2], keypoint[i], color, 2);
		}
		imshow(winName, tmp);
	}

	return 0;
}
