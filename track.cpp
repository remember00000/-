#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include"opencv2/legacy/legacy.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

VideoCapture capture;
Mat frame,model,gray,prevgray;  
Mat img1, img2, out_img;
vector<KeyPoint>keypoints_obj, keypoints_scene;
Mat descriptor_obj, descriptor_scene, maskROI;
vector<Point2f> fpts[2];
vector<uchar> status; // 特征点跟踪成功标志位
vector<float> errors; // 跟踪时候区域误差和
vector<int> lastidx;
int hessian_thr = 500, waittime = 1;
float knnratio = 0.7;
int outflag = 1;	string outputVideoPath = "D:\\CV\\ex12\\match_rancsac.avi";

vector<DMatch> goodmatches, goodmatches2;

vector<DMatch>& findMatch(Mat& model, Mat& frame, vector<KeyPoint>keypoints_obj, vector<KeyPoint> keypoints_scene, Mat& descriptor_obj, Mat& descriptor_scene, Mat& out_img) ;
/*
Mat& select(Mat& img) {
	Rect box= selectROI("select", img);
	Mat area = img(box);
	return area;
}
*/

Point p0, p1;
string maskwin = "maskMark";
bool left = false;
void onmouse(int event, int x, int y, int flags, void*) {
	switch (event) {
	case EVENT_LBUTTONDOWN:
	{
		p0 = Point(x, y);
		break;
	}
	case EVENT_LBUTTONUP:
	{
		p1 = Point(x, y);
		Rect rect(p0, p1);
		//imwrite("D:/CV/ex12/model.jpg", model(rect));
		rectangle(model, p0, p1, Scalar(0, 0, 255), 2);

		//waitKey(0);
		//cout << maskROI<< "===========\n";
		(maskROI(rect)).setTo(Scalar(1));
		imshow(maskwin, model);
	}
	}
}
String window_name = "Capture - face detection";
vector<KeyPoint> matchedKps;

int main() {//摄像头使用设置参数然后才能使用
	int ray = 1;
	int surf = 1;
	// 实例
	VideoCapture camera;
	VideoWriter outputVideo;
	camera.open(0);    // 打开摄像头, 默认摄像头cameraIndex=0
	if (!camera.isOpened())
	{
		cerr << "Couldn't open camera." << endl;
	}

	// 设置参数
	camera.set(CV_CAP_PROP_FRAME_WIDTH, 1000);      // 宽度
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 1000);    // 高度
	camera.set(CV_CAP_PROP_FPS, 40);                     // 帧率

	// 查询参数
	double frameWidth = camera.get(CV_CAP_PROP_FRAME_WIDTH);
	double frameHeight = camera.get(CV_CAP_PROP_FRAME_HEIGHT);
	double fps = camera.get(CV_CAP_PROP_FPS);
	int idx = 0;
	cout << " fps:" <<fps <<"frameWidth:"<< frameWidth<<"frameheight:"<< frameHeight<<"\n";
		SurfFeatureDetector detector(hessian_thr);//数hessianThreshold是图像Hessian矩阵判别式的阈值，值越大检测出的特征点就越少，也就意味着特征点越稳定。
		SurfDescriptorExtractor SurfDescriptor;
		//SiftFeatureDetector detector(hessian_thr);//数hessianThreshold是图像Hessian矩阵判别式的阈值，值越大检测出的特征点就越少，也就意味着特征点越稳定。
		//SiftDescriptorExtractor SiftDescriptor;

	outputVideo.open(outputVideoPath, CV_FOURCC('D', 'I', 'V', 'X'), 35, Size(frameWidth*2, frameHeight));
	vector<DMatch> select;
	// 循环读取视频帧
	bool needflush = false;//标志使用surf重定位
	while (true)
	{
		
		camera >> frame;
		cvtColor(frame, gray, CV_BGR2GRAY);
		//cout << frame.size() << "******\n";
		if (idx == 0) {//第一帧选取ROI
			if (ray) {
				prevgray = gray.clone();
			}

			idx++;
			frame.copyTo(model);
			imshow(maskwin, model);
			maskROI = Mat::zeros(model.rows, model.cols, CV_8UC1);
			//cout << maskROI.size() << "+++++++++++\n";
			setMouseCallback(maskwin, onmouse);
			if (waitKey(0) == ' ') {
				detector.detect(gray, keypoints_obj, maskROI);
				SurfDescriptor.compute(gray, keypoints_obj, descriptor_obj);
				if(ray) KeyPoint::convert(keypoints_obj, fpts[0]);//光流使用将特征点转成point存储 用于追踪
				cout <<"Initial:"<<fpts[0].size() << "*******************\n";
				continue;
			};

			//maskROI.create(frame.size(), CV_8UC1);
			//maskROI.setTo(0);	
		}
		
		if (needflush||idx%5==1) {//每5帧使用surf或者强制使用surf
			needflush = false;
			//cout << "sift\n";
			detector.detect(gray, keypoints_scene);
			//SiftDescriptor.compute(gray, keypoints_scene, descriptor_scene);
			SurfDescriptor.compute(gray, keypoints_scene, descriptor_scene);
			goodmatches.clear();
			findMatch(model, frame, keypoints_obj, keypoints_scene, descriptor_obj, descriptor_scene, out_img);
			if (ray) {//使用光流
				//只对匹配上的点进行光流追踪
				matchedKps.clear();
				lastidx.clear();
				for (int i = 0; i < goodmatches.size(); i++) {
					matchedKps.push_back(keypoints_scene[goodmatches[i].trainIdx]);
					lastidx.push_back(goodmatches[i].trainIdx);//记录追踪点的特征索引
				}
				fpts[0].clear();
				//KeyPoint::convert(keypoints_scene, fpts[0]);
				for (int i = 0; i < matchedKps.size(); i++) {//不能改变kp，利用索引查找
					fpts[0].push_back(matchedKps[i].pt);
				}
			}
			prevgray = gray.clone();//光流记录上一次的灰度图
			if (fpts[0].size() == 0) {
				needflush = true;//没有追踪点重新查找特征点
				continue;
			}
		}
	 else {
			calcOpticalFlowPyrLK(prevgray, gray, fpts[0], fpts[1], status, errors);//光流追踪
			if (fpts[1].size() == 0) {//使用durf重新查找
				needflush = true;
				continue;
			}
			for (int i = 0; i < fpts[1].size(); i++) {
				keypoints_scene[lastidx[i]].pt=fpts[1][i];//光流追踪后的特征点刷新在特征点中的point坐标
			}
			swap(fpts[0], fpts[1]);//指针重新记录本次特征点位置
			prevgray = gray.clone();
		}
		idx++;
		drawMatches(model, keypoints_obj, frame, keypoints_scene, goodmatches, out_img,
				Scalar(0, 0, 255), Scalar(0, 0, 255), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);//绘制匹配点
		if(outflag) outputVideo.write(out_img);//是否输出视频
		imshow(window_name, out_img);
		if (waitKey(waittime) == 'q') break;   // ESC 键退出
	}
	// 释放
	camera.release();
	destroyAllWindows();
	return 0;
}
vector<DMatch>& findMatch(Mat& model, Mat& frame, vector<KeyPoint>keypoints_obj, vector<KeyPoint> keypoints_scene, Mat& descriptor_obj, Mat& descriptor_scene, Mat& out_img) {
	FlannBasedMatcher fbmatcher;
	vector<DMatch> matches;
	//将找到的描述子进行匹配并存入matches中
	vector<vector<DMatch>> knnmatches;
	fbmatcher.knnMatch( descriptor_obj, descriptor_scene, knnmatches,2, Mat());//找到最近的两个
	for (int i = 0; i < knnmatches.size(); i++) {
		cout << knnmatches[i][0].distance << "----\n";
		if (knnmatches[i][0].distance < knnratio * knnmatches[i][1].distance) {//通过前两个最优的距离比，筛选找到的匹配点
			goodmatches.push_back(knnmatches[i][0]);
		}
	}
	return goodmatches;
	/* match方法与ransac
	
	fbmatcher.match(descriptor_obj, descriptor_scene, matches, Mat());
	double minDist1 = 1000;
	double maxDist1 = 0;
	//找出最优描述子
	for (int i = 0; i < descriptor_obj.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist1)
		{
			minDist1 = dist;
		}
		if (dist > maxDist1)
		{
			maxDist1 = dist;
		}

	}
	//通过最值距离筛选匹配点
	//cout << "mindist:" << minDist1 << "maxdist:" << maxDist1 << "\n";
	for (int i = 0; i < descriptor_obj.rows; i++)
	{
		double dist = matches[i].distance;
		//minDist1+(minDist1+maxDist1)/2     max(2 * minDist1, 0.02)   maxDist1/2
		if (dist < max(2 * minDist1, 0.02))//可以调参
		{
			goodmatches.push_back(matches[i]);
		}
	}
	//cout << "goodmatches.size: " << goodmatches.size() << "\n";
	if (goodmatches.size()>10) {//ransac筛选
		vector<Point2f> model_good, scene_good;
		for (int i = 0; i < goodmatches.size(); i++) {
			model_good.push_back(keypoints_obj[goodmatches[i].queryIdx].pt);//模板中的特征点
			scene_good.push_back(keypoints_scene[goodmatches[i].trainIdx].pt);//图像上的匹配点
		}
		//cout << "model_good: " << model_good.size() << "scene_good: " << scene_good.size() << "\n";
		vector<unsigned char> listpoints;
		Mat H = findHomography(model_good, scene_good, CV_RANSAC, 3, listpoints);//单应性矩阵函数中的rnasac方法
		goodmatches2.clear();
		for (int i = 0; i < listpoints.size(); i++) {
			if ((int)listpoints[i]) {
				goodmatches2.push_back(goodmatches[i]);

			}
		}
		//cout << "goood2:" << goodmatches2.size() << "\n";
		if (goodmatches2.size() < 10) goodmatches2.clear();
		return goodmatches2;
	}
	else {
		return goodmatches;
	}
	*/
}



void ORB_demo(int, void*)
{
	int Hession = 400;
	double t1 = getTickCount();
	//特征点提取
	//Ptr<ORB> detector = ORB::create(400);
	SurfFeatureDetector detector(5000);
	vector<KeyPoint> keypoints_obj;
	vector<KeyPoint> keypoints_scene;
	//定义描述子
	Mat descriptor_obj, descriptor_scene;
	//检测并计算成描述子
	//detector->detectAndCompute(img1, keypoints_obj);
	//detector->detect(img2, keypoints_scene);
	detector.detect(img1, keypoints_obj);
	detector.detect(img2, keypoints_scene);

	SurfDescriptorExtractor SurfDescriptor;
	SurfDescriptor.compute(img1, keypoints_obj, descriptor_obj);
	SurfDescriptor.compute(img2, keypoints_scene, descriptor_scene);

	//double t = (t2 - t1) * 1000 / getTickFrequency();
	if (descriptor_obj.type() != CV_32F || descriptor_scene.type() != CV_32F)
	{
		descriptor_obj.convertTo(descriptor_obj, CV_32F);
		descriptor_scene.convertTo(descriptor_scene, CV_32F);
	}
	//特征匹配
	//cv::Ptr<cv::DescriptorMatcher> fbmatcher = cv::DescriptorMatcher::create("FlannBased");
	FlannBasedMatcher fbmatcher;
	vector<DMatch> matches;
	//将找到的描述子进行匹配并存入matches中
	fbmatcher.match(descriptor_obj, descriptor_scene, matches,Mat());

	double minDist1 = 1000;
	double maxDist1 = 0;
	//找出最优描述子
	vector<DMatch> goodmatches;
	for (int i = 0; i < descriptor_obj.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist1)
		{
			minDist1 = dist;
		}
		if (dist > maxDist1)
		{
			maxDist1 = dist;
		}

	}
	for (int i = 0; i < descriptor_obj.rows; i++)
	{
		double dist = matches[i].distance;
		//minDist1+(minDist1+maxDist1)/2   max(2 * minDist1, 0.02)
		if (dist < max(2 * minDist1, 0.02))//可以调参
		{
			goodmatches.push_back(matches[i]);
		}
	}
	Mat orbImg;

	drawMatches(img1, keypoints_obj, img2, keypoints_scene, goodmatches, orbImg,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	/*
	cout << "ORB执行时间为:" << t << "ms" << endl;
	cout << "最小距离为：" << minDist1 << endl;
	cout << "最大距离为：" << maxDist1 << endl;
	*/
	imshow("ORB_demo", orbImg);
}
void save_model() {

}

/*
	//----------目标物体用矩形标识出来------------
	vector<Point2f> obj;
	vector<Point2f>scene;
	for (size_t i = 0; i < goodmatches.size(); i++)
	{
		obj.push_back(keypoints_obj[goodmatches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[goodmatches[i].trainIdx].pt);
	}
	vector<Point2f> obj_corner(4);
	vector<Point2f> scene_corner(4);
	//生成透视矩阵
	Mat H = findHomography(obj, scene, RANSAC);

	obj_corner[0] = Point(0, 0);
	obj_corner[1] = Point(img1.cols, 0);
	obj_corner[2] = Point(img1.cols, img1.rows);
	obj_corner[3] = Point(0, img1.rows);
	//透视变换
	perspectiveTransform(obj_corner, scene_corner, H);
	Mat resultImg = orbImg.clone();


	for (int i = 0; i < 4; i++)
	{
		line(resultImg, scene_corner[i] + Point2f(img1.cols, 0), scene_corner[(i + 1) % 4] + Point2f(img1.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("result image", resultImg);

*/

