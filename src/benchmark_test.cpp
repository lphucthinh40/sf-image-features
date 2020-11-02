/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    vector<string> detectorTypes   = {"HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<string> descriptorTypes = {"BRIEF", "ORB", "FREAK", "AKAZE", "SIFT", "BRISK"};

    fstream fout;
    fout.open("../experiment.csv", ios::out | ios::app); 
    fout << "detector,descriptor,image_id,n_keypoint,n_match,t_detector,t_descriptor" << endl;

    /* MAIN LOOP OVER ALL IMAGES */
    for (auto des : descriptorTypes) 
    {   for (auto det : detectorTypes) 
        {   
            cout << "Detector: " << det << " | Descriptor: " << des << endl;
            if (des == "AKAZE" && det != "AKAZE") // AKAZE descriptor only works with AKAZE detector
                continue;
            
            // vector to count keypoints
            vector<int> keypoint_counts;
            vector<int> match_counts;
            vector<double> detect_elapsedTime;
            vector<double> descript_elapsedTime;
            double temp_elapsedTime;

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                if (dataBuffer.size() == dataBufferSize)
                    dataBuffer.erase(dataBuffer.begin());
                dataBuffer.push_back(frame);

                /* DETECT IMAGE KEYPOINTS */
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                string detectorType = det;

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, temp_elapsedTime, false);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, temp_elapsedTime, false);
                }
                else
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, temp_elapsedTime, false);
                }
                detect_elapsedTime.push_back(temp_elapsedTime);
                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                bool bInside = false;
                cv::Rect vehicleRect(535, 180, 180, 150);

                if (bFocusOnVehicle)
                {
                    for (auto it=keypoints.begin(); it != keypoints.end(); )
                    {
                        if (vehicleRect.contains(it->pt))    
                        {    
                            ++it;
                        }
                        else
                        {
                            it = keypoints.erase(it);   // remove outsiders
                        }
                    }
                }

                keypoint_counts.push_back(keypoints.size());

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;

                /* EXTRACT KEYPOINT DESCRIPTORS */
                cv::Mat descriptors;
                string descriptorType = des; // BRIEF, ORB, FREAK, AKAZE, SIFT
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, temp_elapsedTime);

                descript_elapsedTime.push_back(temp_elapsedTime);

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {
                    /* MATCH KEYPOINT DESCRIPTORS */
                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    string distanceType = (descriptorType.compare("SIFT") == 0)? "DES_HOG":"DES_BINARY"; // DES_BINARY, DES_HOG
                    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
           
                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                     (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                     matches, distanceType, matcherType, selectorType);

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;
                    
                    match_counts.push_back(matches.size());
                }
                else
                match_counts.push_back(0);

            } // eof loop over all images
            
            // display to std output
            for (int i=0; i<10; ++i)
            {
                cout << i << ", " << keypoint_counts[i] << ", " << match_counts[i] << ", " 
                     << fixed << setprecision(6) << detect_elapsedTime[i] << ", " << descript_elapsedTime[i] << endl;
            }
            // write to csv file
            for (int i=0; i<10; ++i)
            {
                fout << det << "," << des << "," << i << "," << keypoint_counts[i] << "," << match_counts[i] << "," 
                     << fixed << setprecision(6) << detect_elapsedTime[i] << "," << descript_elapsedTime[i] << endl;
            }
            dataBuffer.clear();
        }       
    }
    fout.close();
    return 0;
}
