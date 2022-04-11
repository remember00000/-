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
vector<uchar> status; // ��������ٳɹ���־λ
vector<float> errors; // ����ʱ����������
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

int main() {//����ͷʹ�����ò���Ȼ�����ʹ��
	int ray = 1;
	int surf = 1;
	// ʵ��
	VideoCapture camera;
	VideoWriter outputVideo;
	camera.open(0);    // ������ͷ, Ĭ������ͷcameraIndex=0
	if (!camera.isOpened())
	{
		cerr << "Couldn't open camera." << endl;
	}

	// ���ò���
	camera.set(CV_CAP_PROP_FRAME_WIDTH, 1000);      // ���
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 1000);    // �߶�
	camera.set(CV_CAP_PROP_FPS, 40);                     // ֡��

	// ��ѯ����
	double frameWidth = camera.get(CV_CAP_PROP_FRAME_WIDTH);
	double frameHeight = camera.get(CV_CAP_PROP_FRAME_HEIGHT);
	double fps = camera.get(CV_CAP_PROP_FPS);
	int idx = 0;
	cout << " fps:" <<fps <<"frameWidth:"<< frameWidth<<"frameheight:"<< frameHeight<<"\n";
		SurfFeatureDetector detector(hessian_thr);//��hessianThreshold��ͼ��Hessian�����б�ʽ����ֵ��ֵԽ��������������Խ�٣�Ҳ����ζ��������Խ�ȶ���
		SurfDescriptorExtractor SurfDescriptor;
		//SiftFeatureDetector detector(hessian_thr);//��hessianThreshold��ͼ��Hessian�����б�ʽ����ֵ��ֵԽ��������������Խ�٣�Ҳ����ζ��������Խ�ȶ���
		//SiftDescriptorExtractor SiftDescriptor;

	outputVideo.open(outputVideoPath, CV_FOURCC('D', 'I', 'V', 'X'), 35, Size(frameWidth*2, frameHeight));
	vector<DMatch> select;
	// ѭ����ȡ��Ƶ֡
	bool needflush = false;//��־ʹ��surf�ض�λ
	while (true)
	{
		
		camera >> frame;
		cvtColor(frame, gray, CV_BGR2GRAY);
		//cout << frame.size() << "******\n";
		if (idx == 0) {//��һ֡ѡȡROI
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
				if(ray) KeyPoint::convert(keypoints_obj, fpts[0]);//����ʹ�ý�������ת��point�洢 ����׷��
				cout <<"Initial:"<<fpts[0].size() << "*******************\n";
				continue;
			};

			//maskROI.create(frame.size(), CV_8UC1);
			//maskROI.setTo(0);	
		}
		
		if (needflush||idx%5==1) {//ÿ5֡ʹ��surf����ǿ��ʹ��surf
			needflush = false;
			//cout << "sift\n";
			detector.detect(gray, keypoints_scene);
			//SiftDescriptor.compute(gray, keypoints_scene, descriptor_scene);
			SurfDescriptor.compute(gray, keypoints_scene, descriptor_scene);
			goodmatches.clear();
			findMatch(model, frame, keypoints_obj, keypoints_scene, descriptor_obj, descriptor_scene, out_img);
			if (ray) {//ʹ�ù���
				//ֻ��ƥ���ϵĵ���й���׷��
				matchedKps.clear();
				lastidx.clear();
				for (int i = 0; i < goodmatches.size(); i++) {
					matchedKps.push_back(keypoints_scene[goodmatches[i].trainIdx]);
					lastidx.push_back(goodmatches[i].trainIdx);//��¼׷�ٵ����������
				}
				fpts[0].clear();
				//KeyPoint::convert(keypoints_scene, fpts[0]);
				for (int i = 0; i < matchedKps.size(); i++) {//���ܸı�kp��������������
					fpts[0].push_back(matchedKps[i].pt);
				}
			}
			prevgray = gray.clone();//������¼��һ�εĻҶ�ͼ
			if (fpts[0].size() == 0) {
				needflush = true;//û��׷�ٵ����²���������
				continue;
			}
		}
	 else {
			calcOpticalFlowPyrLK(prevgray, gray, fpts[0], fpts[1], status, errors);//����׷��
			if (fpts[1].size() == 0) {//ʹ��durf���²���
				needflush = true;
				continue;
			}
			for (int i = 0; i < fpts[1].size(); i++) {
				keypoints_scene[lastidx[i]].pt=fpts[1][i];//����׷�ٺ��������ˢ�����������е�point����
			}
			swap(fpts[0], fpts[1]);//ָ�����¼�¼����������λ��
			prevgray = gray.clone();
		}
		idx++;
		drawMatches(model, keypoints_obj, frame, keypoints_scene, goodmatches, out_img,
				Scalar(0, 0, 255), Scalar(0, 0, 255), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);//����ƥ���
		if(outflag) outputVideo.write(out_img);//�Ƿ������Ƶ
		imshow(window_name, out_img);
		if (waitKey(waittime) == 'q') break;   // ESC ���˳�
	}
	// �ͷ�
	camera.release();
	destroyAllWindows();
	return 0;
}
vector<DMatch>& findMatch(Mat& model, Mat& frame, vector<KeyPoint>keypoints_obj, vector<KeyPoint> keypoints_scene, Mat& descriptor_obj, Mat& descriptor_scene, Mat& out_img) {
	FlannBasedMatcher fbmatcher;
	vector<DMatch> matches;
	//���ҵ��������ӽ���ƥ�䲢����matches��
	vector<vector<DMatch>> knnmatches;
	fbmatcher.knnMatch( descriptor_obj, descriptor_scene, knnmatches,2, Mat());//�ҵ����������
	for (int i = 0; i < knnmatches.size(); i++) {
		cout << knnmatches[i][0].distance << "----\n";
		if (knnmatches[i][0].distance < knnratio * knnmatches[i][1].distance) {//ͨ��ǰ�������ŵľ���ȣ�ɸѡ�ҵ���ƥ���
			goodmatches.push_back(knnmatches[i][0]);
		}
	}
	return goodmatches;
	/* match������ransac
	
	fbmatcher.match(descriptor_obj, descriptor_scene, matches, Mat());
	double minDist1 = 1000;
	double maxDist1 = 0;
	//�ҳ�����������
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
	//ͨ����ֵ����ɸѡƥ���
	//cout << "mindist:" << minDist1 << "maxdist:" << maxDist1 << "\n";
	for (int i = 0; i < descriptor_obj.rows; i++)
	{
		double dist = matches[i].distance;
		//minDist1+(minDist1+maxDist1)/2     max(2 * minDist1, 0.02)   maxDist1/2
		if (dist < max(2 * minDist1, 0.02))//���Ե���
		{
			goodmatches.push_back(matches[i]);
		}
	}
	//cout << "goodmatches.size: " << goodmatches.size() << "\n";
	if (goodmatches.size()>10) {//ransacɸѡ
		vector<Point2f> model_good, scene_good;
		for (int i = 0; i < goodmatches.size(); i++) {
			model_good.push_back(keypoints_obj[goodmatches[i].queryIdx].pt);//ģ���е�������
			scene_good.push_back(keypoints_scene[goodmatches[i].trainIdx].pt);//ͼ���ϵ�ƥ���
		}
		//cout << "model_good: " << model_good.size() << "scene_good: " << scene_good.size() << "\n";
		vector<unsigned char> listpoints;
		Mat H = findHomography(model_good, scene_good, CV_RANSAC, 3, listpoints);//��Ӧ�Ծ������е�rnasac����
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
	//��������ȡ
	//Ptr<ORB> detector = ORB::create(400);
	SurfFeatureDetector detector(5000);
	vector<KeyPoint> keypoints_obj;
	vector<KeyPoint> keypoints_scene;
	//����������
	Mat descriptor_obj, descriptor_scene;
	//��Ⲣ�����������
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
	//����ƥ��
	//cv::Ptr<cv::DescriptorMatcher> fbmatcher = cv::DescriptorMatcher::create("FlannBased");
	FlannBasedMatcher fbmatcher;
	vector<DMatch> matches;
	//���ҵ��������ӽ���ƥ�䲢����matches��
	fbmatcher.match(descriptor_obj, descriptor_scene, matches,Mat());

	double minDist1 = 1000;
	double maxDist1 = 0;
	//�ҳ�����������
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
		if (dist < max(2 * minDist1, 0.02))//���Ե���
		{
			goodmatches.push_back(matches[i]);
		}
	}
	Mat orbImg;

	drawMatches(img1, keypoints_obj, img2, keypoints_scene, goodmatches, orbImg,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	/*
	cout << "ORBִ��ʱ��Ϊ:" << t << "ms" << endl;
	cout << "��С����Ϊ��" << minDist1 << endl;
	cout << "������Ϊ��" << maxDist1 << endl;
	*/
	imshow("ORB_demo", orbImg);
}
void save_model() {

}

/*
	//----------Ŀ�������þ��α�ʶ����------------
	vector<Point2f> obj;
	vector<Point2f>scene;
	for (size_t i = 0; i < goodmatches.size(); i++)
	{
		obj.push_back(keypoints_obj[goodmatches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[goodmatches[i].trainIdx].pt);
	}
	vector<Point2f> obj_corner(4);
	vector<Point2f> scene_corner(4);
	//����͸�Ӿ���
	Mat H = findHomography(obj, scene, RANSAC);

	obj_corner[0] = Point(0, 0);
	obj_corner[1] = Point(img1.cols, 0);
	obj_corner[2] = Point(img1.cols, img1.rows);
	obj_corner[3] = Point(0, img1.rows);
	//͸�ӱ任
	perspectiveTransform(obj_corner, scene_corner, H);
	Mat resultImg = orbImg.clone();


	for (int i = 0; i < 4; i++)
	{
		line(resultImg, scene_corner[i] + Point2f(img1.cols, 0), scene_corner[(i + 1) % 4] + Point2f(img1.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("result image", resultImg);

*/

