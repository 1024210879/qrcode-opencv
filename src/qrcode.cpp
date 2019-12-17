#include <string>				// string ��
#include <vector>				// vector ������
#include <cmath>				// ��ѧ ��
#include <opencv2/highgui.hpp>	// opencv �߼� GUI ��
#include <opencv2/imgproc.hpp>	// opencv ͼ������̿�
using namespace std;			// ��׼�����ռ�
using namespace cv;				// opencv �����ռ�

#define PI 3.1415926

// ����ת�������ĵ����
double dist(RotatedRect a, RotatedRect b)
{
	double dx = a.center.x - b.center.x;
	double dy = a.center.y - b.center.y;
	return sqrtf(dx * dx + dy * dy);
}

// �������
double dist(Point2f a, Point2f b)
{
	double dx = a.x - b.x;
	double dy = a.y - b.y;
	return sqrtf(dx * dx + dy * dy);
}

// ������
int main(int argc, char* argv[])
{
	// -------------------------------------------------------------------------------------
	// ----------������������������---------- Ԥ����  ----------������������������----------
	// -------------------------------------------------------------------------------------

	// ��ά��ͼƬ·��
	string srcPath;
	if (argc == 1) srcPath = "pic/9.jpg";
	else srcPath = argv[1];
	
	// ��ͼ ԭͼ��Ϊ src
	Mat src = imread(srcPath);
	if (src.empty()) return 0;				// �ж�ͼ���Ƿ�Ϊ��
	imshow("src", src);
	
	// ת��Ϊ�Ҷ�ͼ ��Ϊ����ͼ img
	Mat img = src.clone();					// ԭͼ���������ͼ
	cvtColor(img, img, COLOR_BGR2GRAY);		// ת�Ҷ�ͼ
	imshow("gray", img);
	
	// ��ֵ�˲� ȥ�������� �˲�ģ��ߴ�(5 * 5)
	medianBlur(img, img, 5);
	imshow("medianBlur", img);				

	// ��ֵ��
	string winName = "threshold | press 'q' to continue";	// ������
	int value = 100;										// ��ֵ����ֵ
	Mat tmp = Mat::zeros(src.size(), img.type());			// tmp ͼ�����ڻ��ƹ���ͼ��ʱ�洢
	namedWindow(winName, 0);								// ��������
	resizeWindow(winName, Size(tmp.cols, tmp.rows + 45));
	createTrackbar("value", winName, &value, 256);			// ����������
	while (waitKey(33) != 'q')								// ÿ 33 ������һ�ΰ��� �� 'q' ������ѭ��
	{
		threshold(img, tmp, value - 1, 255, THRESH_BINARY);	// ��ֵ��
		imshow(winName, tmp);
	}
	img = tmp.clone();										// ���� tmp �� img

	// tmp ��Ϊ3ͨ�� BGR ͼ�� ���ҿ�ȱ�Ϊ 2 �� ����ͬ����������ͼ�Ա�
	tmp = Mat::zeros(Size(src.cols << 1, img.rows), CV_8UC3);				
	Mat rightPart = tmp(Rect(img.cols, 0, img.cols, img.rows));				// tmp �Ұ벿��
	Mat leftPart = tmp(Rect(0, 0, img.cols, img.rows));						// tmp ��벿��


	// -------------------------------------------------------------------------------------
	// ----------������������������---------- ��λ��  ----------������������������----------
	// -------------------------------------------------------------------------------------

	// ��Ե��⣨����������
	vector<vector<Point>> contours;											// ��¼����
	vector<Point2f> keypoint;												// ��¼��λ����������
	vector<Vec4i> hierarchy;												// ��¼�����㼶��ϵ
	findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);	// opencv �⺯�� ��������
	img = src.clone();														// ����һ��ԭͼ ��������͸����λ�����

	// �� tmp �Ұ벿�ֻ����������� ��ɫ��ʾ
	for (int i = 0; i < contours.size(); ++i)
		drawContours(rightPart, contours, i, Scalar(255, 255, 255));		

	// �� tmp �Ұ벿�����»��ƶ�λ������ ��ɫ��ʾ
	for (int i = 0; i < contours.size(); ++i)
	{
		RotatedRect rect = minAreaRect(Mat(contours[i]));					// ����������С��Χ����
		double width = rect.size.width;										// ��������һ��	
		double height = rect.size.height;									// ��������һ��

		// �жϿ�߱�
		if (min(width, height) / max(width, height) > 0.8)
		{
			// �ж� ����������ֻ��һ��
			int child_1 = hierarchy[i][2];														
			if (child_1 != -1 && hierarchy[child_1][0] == -1 && hierarchy[child_1][1] == -1)	
			{
				// �ж� ������������ֻ��һ��
				int child_2 = hierarchy[child_1][2];											
				if (child_2 != -1 && hierarchy[child_2][0] == -1 && hierarchy[child_2][1] == -1)
				{
					// ��Χ��������������������������С��Χ����
					RotatedRect rect_child_1 = minAreaRect(Mat(contours[child_1]));
					RotatedRect rect_child_2 = minAreaRect(Mat(contours[child_2]));

					// �������ĵ�ľ���
					double dist_1 = dist(rect, rect_child_1);
					double dist_2 = dist(rect_child_1, rect_child_2);
					double dist_3 = dist(rect_child_2, rect);

					// ���ĵ������������Χ
					int offset = 5;	

					// �ж� ��Χ������������������������������һ����
					if (abs(dist_1 - dist_2) < offset && abs(dist_2 - dist_3) < offset &&abs(dist_3 - dist_1) < offset)
					{
						// �����ĸ�����������ж�Ϊ��ά��Ķ�λ��
						// �� img �ϻ��ƶ�λ��
						drawContours(img, contours, i, Scalar(0, 255, 0), -1);
						// �� tmp �Ұ벿�ֻ��ƶ�λ��
						drawContours(rightPart, contours, i, Scalar(0, 255, 0));
						drawContours(rightPart, contours, child_1, Scalar(0, 255, 0));
						drawContours(rightPart, contours, child_2, Scalar(0, 255, 0));
						// ��¼��λ�����������
						keypoint.push_back(Point2f(rect.center.x, rect.center.y));
					}
				}
			}
		}
	}
	
	line(tmp, Point(src.cols, 0), Point(src.cols, src.rows - 1), Scalar(0, 0, 255), 3);	// �� tmp �м仭���� ��������������
	addWeighted(src, 0.5, img, 0.5, 0, img);	// һ�ݻ��ƹ���λ��� �� һ��ԭͼ �� 0.5 ������� �ﵽ��͸��Ч��
	img.copyTo(leftPart);						// copy ����벿��
	imshow("keypoint", tmp);

	// �ͷ��ڴ�
	vector<vector<Point>>().swap(contours);
	vector<Vec4i>().swap(hierarchy);


	// -------------------------------------------------------------------------------------
	// ----------������������������---------- �����  ----------������������������----------
	// -------------------------------------------------------------------------------------

	// ���һ��ͼ������ N ����ά������⵽ N * 3 �����ĵ� ����Ҫ�����Щ���ĵ�
	// ������Щ������⵽ N * 3 �� M �����ĵ� 
	// ���� ĳ��ά��ĳ��λ��δ��⵽
	// ���� ĳ��ά�뱻�����ڵ� ֻ¶��һ������λ��

	int len_k = 85;									// ģ��ֵ
	int angle_low = 75;								// cos ����
	int angle_high = 105;							// cos ����
	winName = "match keypoint";
	namedWindow(winName, 0);
	resizeWindow(winName, Size(img.cols, img.rows + 165));
	createTrackbar("len_k", winName, &len_k, 100);
	createTrackbar("angle_low", winName, &angle_low, 180);
	createTrackbar("angle_high", winName, &angle_high, 180);

	// ��¼���ĵ��������
	vector<vector<double>> d;
	for (int i = 0; i + 1 < keypoint.size(); i++)
	{
		d.push_back(vector<double>(keypoint.size()));
		for (int j = i + 1; j < keypoint.size(); j++)
			d[i][j] = (dist(keypoint[i], keypoint[j]));
	}

	// �� 'q' ����
	while (waitKey(33) != 'q')
	{
		vector<Point2f> keypointDst;					// ��˳���¼��Ե�
		vector<int> contoursMask(keypoint.size(), 0);	// ����Ƿ����

		// ��ʼ���
		for (int i = 0; i + 2 < keypoint.size(); i++)
		{
			if (contoursMask[i]) continue;				// ��������Եĵ�
			for (int j = i + 1; j + 1 < keypoint.size(); j++)
			{
				if (contoursMask[j]) continue;			// ��������Եĵ�
				for (int k = j + 1; k < keypoint.size(); k++)
				{
					if (contoursMask[k]) continue;		// ��������Եĵ�

					// �߳�
					double a = d[i][j], b = d[i][k], c = d[j][k];

					// ����
					if (a > b) swap(a, b); 
					if (a > c) swap(a, c); 
					if (b > c) swap(b, c);

					// �Ƕ� ���Ҷ���
					double A = acos((b * b + c * c - a * a) / (2 * b * c)) / PI * 180;
					double B = acos((a * a + c * c - b * b) / (2 * a * c)) / PI * 180;
					double C = acos((a * a + b * b - c * c) / (2 * a * b)) / PI * 180;

					// �ж��Ƿ���� ĳ���߳�֮������ֵ�� ĳ������ֵ��
					if ((a / b > len_k * 0.01 && angle_low < C && C < angle_high)
						|| (b / c > len_k * 0.01 && angle_low < A && A < angle_high)
						|| (a / c > len_k * 0.01 && angle_low < B && B < angle_high))
					{
						// ���
						contoursMask[i] = 1; contoursMask[j] = 1; contoursMask[k] = 1;

						// ��¼
						keypointDst.push_back(keypoint[i]);
						keypointDst.push_back(keypoint[j]);
						keypointDst.push_back(keypoint[k]);
					}
				}
			}
		}

		// ��Զ�λ������ ��ɫ��ʾ
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
