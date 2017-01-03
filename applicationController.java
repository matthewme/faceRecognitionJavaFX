package finalProject;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javax.imageio.ImageIO;
import javax.swing.JFileChooser;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import cern.colt.Arrays;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import finalProject.ImageIo;
import finalProject.Face;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Tab;
import javafx.scene.control.TextField;
import javafx.scene.control.TitledPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
//SNAPSHOT

/**
 * The controller for our application, where the application logic is
 * implemented. It handles the button for starting/stopping the camera and the
 * acquired video stream.
 * @author Matthew Martinez
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @author <a href="http://max-z.de">Maximilian Zuleger</a> (minor fixes)
 * @version 2.0 (2016-09-17)
 * @since 1.0 (2013-10-20)
 *
 */
public class applicationController
{
	// the FXML button
	@FXML
	private Button captureImage, clrCap, clrLast, imgLoc, start_btn,pre_img_btn,next_img_btn,createTraining;
	@FXML
	private ImageView currentFrame, currentFrameStatic, capImgOne,capImgTwo,capImgThree, capImgFour, capImgFive, capImgSix;
	@FXML
	private ImageView avgImageView, oriImage,eigenImg,frameFaceRecTab,frameCapFaceRecTab,finalFaceImgView;
	@FXML
	private RadioButton grayscale;
	@FXML
	private TitledPane titlePane1;
	@FXML
	private TextField heightBox, widthBox, imgNames,numTrainImgs;
	@FXML
	private ScrollPane trainingScrollPane;
	@FXML
	private Pane eFacesPane;
	@FXML
	private HBox eFacesHBox;
	@FXML
	private Tab trainingTab;
	@FXML
	private Label oriImgNameLbl,oriImgNumLbl,displayNameLbl;
	

	CascadeClassifier faceDetectorClassifier = new CascadeClassifier("D:/matthew/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml");
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that realizes the video capture
	private VideoCapture capture = new VideoCapture();
	// a flag to change the button behavior
	private boolean cameraActive = false;
	// the id of the camera to be used
	private static int cameraId = 0;
	//Load File Directory
	File[] files = new File("D:\\Matthew\\workspace\\finalProject\\students").listFiles();
	
	int ti = tTabHandler(null);//Retrieve the total count of files in the database
	Face[] faces = new Face[ti];//Array of object Face
	
	int num = 0;
	double[][] weights;//[M][numEFaces]
	double[] avgImageGlobal;
	double[][] eigenFacesTransposedGlobal;//[N^2][numEFaces]
	int numEFacesGlobal;
	
	//----------------------------------------CAPTURE IMAGE TAB------------------------------------------//
	//The number of Captured Pictures
	int count;
	//The cropped and scaled image
	Mat newROI = new Mat();
	//Stores the path to save location
	JFileChooser chooser;

	/**
	 * The action triggered by pushing the button on the GUI
	 *
	 * @param event
	 *            the push button event
	 */
	@FXML
	protected void startCamera(ActionEvent event)
	{			
		if (!this.cameraActive)
		{
			// start the video capture
			this.capture.open(cameraId);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(currentFrame, imageToShow);
						updateImageView(frameFaceRecTab, imageToShow);
					}							
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				//update the button content
				this.start_btn.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Impossible to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.start_btn.setText("Stop Camera");
			
			// stop the timer
			this.stopAcquisition();
		}
	}
	
	//Take a picture from the camera
	@FXML
	protected void captureImage(ActionEvent event)
	{
		try
		{
		int h = Integer.parseInt(heightBox.getText());
		int w = Integer.parseInt(widthBox.getText());
		Mat ROI = new Mat();
		
	
			if(heightBox != null && widthBox != null)
			{
				
				Mat frame = new Mat();
				// Now detect the face in the image.
			    // MatOfRect is a special container class for Rect.
				MatOfRect faceDetections = new MatOfRect();
			
				capture.retrieve(frame);
				
				faceDetectorClassifier.detectMultiScale(frame, faceDetections); //detectMultiScale will perform the detection
			    //Draw a bounding box around each face.
			    for (Rect rect : faceDetections.toArray()) {
				    Imgproc.rectangle(frame, 
				    new Point(rect.x, rect.y),   //bottom left
				    new Point(rect.x + rect.width, rect.y + rect.height), //top right 
				    new Scalar(0,255,0)); //RGB colour
				    
				    //This operation extracts the frame within the face recognition box
				    ROI = frame.submat(rect.y + 10, (rect.y + rect.height) - 10, rect.x + 10, (rect.x + rect.width) - 10); 
			     }
			    
			    //After the frame is cropped this function will resize the Image
			    Size sz = new Size(h,w);
			    Imgproc.resize(ROI,newROI,sz);
			    
				//If black/white radio button is selected then capture a gray image
				if (grayscale.isSelected())
				{
					Imgproc.cvtColor(newROI, newROI, Imgproc.COLOR_BGR2GRAY);
					Image imageToShow = Utils.mat2Image(newROI);
					updateImageView(currentFrameStatic, imageToShow);
					System.out.println("Width: " + imageToShow.getWidth() + " Length: " + imageToShow.getHeight());
				}
				else//else capture a color image
				{
					Image imageToShow = Utils.mat2Image(newROI);
					updateImageView(currentFrameStatic, imageToShow);
	
					System.out.println("Width: " + imageToShow.getWidth() + " Length: " + imageToShow.getHeight());
				}	
			}
		}
		catch(Exception e)
		{
			System.out.println("Must enter a Length and Width");
		}
	}
	
	@FXML
	protected void addCaptureImage(ActionEvent event)
	{
		Image imageToShow = Utils.mat2Image(newROI);
		
		if(count == 0 && imageToShow != null)
		{
			updateImageView(capImgOne, imageToShow);
			count++;
			System.out.println("The count is: " + count);
		}
		else if(count == 1)
		{
			updateImageView(capImgTwo, imageToShow);
			count++;
			System.out.println("The count is: " + count);
		}
		else if(count == 2)
		{
			updateImageView(capImgThree, imageToShow);
			count++;
			System.out.println("The count is: " + count);
		}
		else if(count == 3)
		{
			updateImageView(capImgFour, imageToShow);
			count++;
			System.out.println("The count is: " + count);
		}
		else if(count == 4)
		{
			updateImageView(capImgFive, imageToShow);
			count++;
			System.out.println("The count is: " + count);
		}
		else if(count == 5)
		{
			updateImageView(capImgSix, imageToShow);
			count++;
			System.out.println("The count is: " + count);
		}
		else
		{
			System.out.println("\nBox is full or ImageView is Null");
		}
	}
	
	@FXML
	protected void clrLast(ActionEvent event)
	{
		if(count == 0)
		{
			System.out.println("Box is Empty");
		}
	    else if(count == 1)
		{
	    	capImgOne.setImage(null);
			count--;
			System.out.println("The count is: " + count);
		}
		else if(count == 2)
		{
			capImgTwo.setImage(null);
			count--;
			System.out.println("The count is: " + count);
		}
		else if(count == 3)
		{
			capImgThree.setImage(null);
			count--;
			System.out.println("The count is: " + count);
		}
		else if(count == 4)
		{
			capImgFour.setImage(null);
			count--;
			System.out.println("The count is: " + count);
		}
		else if(count == 5)
		{
			capImgFive.setImage(null);
			count--;
			System.out.println("The count is: " + count);
		}
		else if(count == 6)
		{
			capImgSix.setImage(null);
			count--;
			System.out.println("The count is: " + count);
		}
	}
	
	@FXML
	protected void clearAllCap(ActionEvent event)
	{
		//Remove All Images
		count = 0;
		capImgOne.setImage(null);
		capImgTwo.setImage(null);
		capImgThree.setImage(null);
		capImgFour.setImage(null);
		capImgFive.setImage(null);
		capImgSix.setImage(null);
		
		System.out.println("The count is: " + count);
	}
	
	@FXML
	protected void locationChooser(ActionEvent event)
	{

		   String choosertitle = "Choose Image Storage Directory";
		   
		    chooser = new JFileChooser(); 
		    chooser.setCurrentDirectory(new java.io.File("."));
		    chooser.setDialogTitle(choosertitle);
		    chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

		    // disable the "All files" option.

		    chooser.setAcceptAllFileFilterUsed(false);
		    //    
		    if (chooser.showOpenDialog(chooser) == JFileChooser.APPROVE_OPTION) { 
		      System.out.println("getCurrentDirectory(): " 
		         +  chooser.getCurrentDirectory());
		      System.out.println("getSelectedFile() : " 
		         +  chooser.getSelectedFile());
		      }
		    else {
		      System.out.println("No Selection ");
		      } 
	}
	
	@FXML
	protected void saveImgs(ActionEvent event)
	{
		//The name of the image
		String name = imgNames.getText();
		
		try
		{
		if(count >= 1)
		{
			//Get the image from the image view
			Image img1 = capImgOne.getImage();
			//Convert Image to buffered image
			BufferedImage ImageToWrite = SwingFXUtils.fromFXImage(img1, null);
			//Save the Image
			saveImage(ImageToWrite, chooser.getSelectedFile()+ "\\"  + name + "1" + ".png");
		}
	    if(count >= 2)
		{
			Image img2 = capImgTwo.getImage();
			BufferedImage ImageToWrite = SwingFXUtils.fromFXImage(img2, null);
			saveImage(ImageToWrite,chooser.getSelectedFile()+ "\\" + name + "2" + ".png");
		}
		if(count >= 3)
		{
			Image img3 = capImgThree.getImage();
			BufferedImage ImageToWrite = SwingFXUtils.fromFXImage(img3, null);
			saveImage(ImageToWrite,chooser.getSelectedFile()+ "\\" + name + "3" + ".png");
		}
	    if(count >= 4)
		{
			Image img4 = capImgFour.getImage();
			BufferedImage ImageToWrite = SwingFXUtils.fromFXImage(img4, null);
			saveImage(ImageToWrite,chooser.getSelectedFile() + "\\" + name + "4" + ".png");
		}
		if(count >= 5)
		{
			Image img5 = capImgFive.getImage();
			BufferedImage ImageToWrite = SwingFXUtils.fromFXImage(img5, null);
			saveImage(ImageToWrite,chooser.getSelectedFile() + "\\" + name + "5" + ".png");
		}
	    if(count >= 6)
		{
			Image img6 = capImgSix.getImage();
			BufferedImage ImageToWrite = SwingFXUtils.fromFXImage(img6, null);
			saveImage(ImageToWrite,chooser.getSelectedFile() + "\\" + name + "6" + ".png");
		}
		}
		catch (Exception e){
			System.out.println("No File Chosen");
		}
	}
	
	  private void saveImage(BufferedImage im, String fnm)
	  // save image in fnm
	  {
	    try {
	      ImageIO.write(im, "png", new File(fnm));
	      System.out.println("Saved image to " + fnm);
	    } 
	    catch (IOException e) {
	      System.out.println("Could not save image to " + fnm);
	    }
	  }  // end of saveImage()
	
	/**
	 * Get a frame from the opened video stream (if any)
	 *
	 * @return the {@link Mat} to show
	 */
	private Mat grabFrame()
	{
		// init everything
		Mat frame = new Mat();
		MatOfRect faceDetections = new MatOfRect();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				
			    faceDetectorClassifier.detectMultiScale(frame, faceDetections); //detectMultiScale will perform the detection
			    // Draw a bounding box around each face.
		        for (Rect rect : faceDetections.toArray()) {
				   Imgproc.rectangle(frame, 
				    new Point(rect.x, rect.y),   //bottom left
				    new Point(rect.x + rect.width, rect.y + rect.height), //top right 
				    new Scalar(255, 0, 0)); //RGB colour
		         }
								
				//if the frame is not empty, process it
				if (!frame.empty())
				{		        
					//Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
					if (grayscale.isSelected())
					{
					   Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
					}			
				}					
			}
			catch (Exception e)
			{
				// log the error
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		
		return frame;
	}
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}

//-----------------------------------------TRAINING IMAGE TAB-------------------------------------------------//
	@FXML
	public void createTraning(ActionEvent event)
	{
		try
		{
		int numEFaces = Integer.parseInt(numTrainImgs.getText());
		if(numEFaces > 0 && numEFaces <= ti)
		{			
			numEFacesGlobal = numEFaces;
		    eFacesHBox.getChildren().clear();	
			Face[] eigenFacesArray = new Face[numEFaces];//Array of EigenFaces
			ImageView[] imageViewArray = new ImageView[numEFaces];
			//Read The files from the Directory
			Utils.showFiles(files,0,faces,0);
			     
		    int h = faces[0].bImg.getHeight();//Height
		    int w = faces[0].bImg.getWidth();//Width
		    int hw = h*w;
		
			//Store All Images in 2 dimensional array.Every row is a column
			double[][] originalImg = new double[ti][hw];
			
			for(int i = 0; i < ti; i++)
			{
				for(int j = 0; j < hw;j++)
				{
					originalImg[i][j] =  faces[i].imageData[j]; 
				}
			}
			
			//------------------------------This section for displaying average Image---------------//
			double[] avgImage = new double[h*w];
			double avg = 0;
			double sum = 0;
			
			//get the average of the images by column
			for(int i = 0; i < (h*w);i++)
			{
				for(int j = 0; j < originalImg.length;j++)
				{
					sum += originalImg[j][i];
				}
				
				avg = sum / originalImg.length;
				avgImage[i] = avg; 
				sum = 0;
			}	
			
			//Turn the avg image into a 2D array for Imageview
			byte[][] blurryImg = new byte[(int) Math.sqrt(hw)][(int) Math.sqrt(hw)];
			//1d to 2d
			for(int j = 0; j < hw;j++)
			{
				int row = j/(int) Math.sqrt(hw);
				int col = j%(int) Math.sqrt(hw);
				
				blurryImg[row][col] = (byte) avgImage[j];
			}
			
			//Display the Average Image
			BufferedImage outImage1 = ImageIo.setGrayByteImageArray2DToBufferedImage(blurryImg);
			Image averageImage = SwingFXUtils.toFXImage(outImage1, null);
			updateImageView(avgImageView, averageImage);
			//-------------------------------------end display average image---------------------//
			//Normalize the original image data
			double[][] originalImgNormalized = new double[ti][hw];
		    double[] mvals = new double[originalImgNormalized.length];
		    
		    originalImgNormalized = originalImg;
		    
		    for (int i = 0; i < originalImgNormalized.length; i++)
		      mvals[i] = Utils.max(originalImgNormalized[i]);

		    for (int i = 0; i < originalImgNormalized.length; i++) {
		      for (int j = 0; j < originalImgNormalized[0].length; j++)
		    	  originalImgNormalized[i][j] /= mvals[i];
		    }
		    		  	   
			//Calculate the normalized average
			double[] avgImgNormalized = new double[h*w];
			avg = 0;
			sum = 0;
			
			//get the average of the images by column
			for(int i = 0; i < (h*w);i++)
			{
				for(int j = 0; j < originalImgNormalized.length;j++)
				{
					sum += originalImg[j][i];
				}
				avg = sum / originalImgNormalized.length;
				avgImgNormalized[i] = avg; 
				sum = 0;
			}
			
			avgImageGlobal = avgImgNormalized;	

		    //Get the training data by subtracting the normalized average from the normalized originals
		    double[][] trainingImgsNormalized = new double[ti][hw]; 
		      
			for(int i = 0; i < ti;i++)
			{
				for(int j = 0; j < hw;j++)
				{
					trainingImgsNormalized[i][j] = originalImgNormalized[i][j] - avgImgNormalized[j];
				}
			}	
		    
			//Calculate covariance Matrix
			DoubleMatrix2D correlationMatrix = Utils.calcCovarMat(trainingImgsNormalized);
			EigenvalueDecomposition eigenDecomp = new EigenvalueDecomposition(correlationMatrix);
			//Get The eigen Vectors
			DoubleMatrix2D eigenVectorMatrix = eigenDecomp.getV();
			double[][] myVectors = eigenVectorMatrix.toArray();
			//Get the eigen Values
			DoubleMatrix1D eigenValues = eigenDecomp.getRealEigenvalues();
			double[] myEigenValues = eigenValues.toArray();
			
//			System.out.println("The Eigenvalues are: " + Arrays.toString(myEigenValues));
				
//			System.out.println("The vectors are: ");
//			for(double[] vector: myVectors)
//			{
//				System.out.println("Rows of eigenvectors " + Arrays.toString(vector));
//			}
			// sort Eigenvectors and Eigenvars into descending order	
			Utils.sortEigenInfo(myEigenValues, myVectors);
			
			//Create the eigen faces. each row is a face
			double[][] eigenFaces = new double[numEFaces][hw];
		    for(int i=0; i< numEFaces; i++)
		    {
		        for(int j=0; j<hw; j++)
		        {
		          eigenFaces[i][j]=0;  
		          for(int k=0; k< numEFaces; k++)
		          {
		        	  eigenFaces[i][j] += myVectors[i][k]*trainingImgsNormalized[k][j];
		          }
		        }
		    }
		    
		    //scale the eigenfaces  to 255 and convert to byte. Each row is a face
		    byte[][] eigenFacesScaled = new byte[numEFaces][hw];
	        double min =eigenFaces[0][0];
	        double max =eigenFaces[0][0];
	        
	        for(int i = 0; i < numEFaces; i++)
	        {
		        for (int j = 0; j < hw; j++)
		        {
		            if( eigenFaces[i][j] >max )
		                max = eigenFaces[i][j];
		            if( eigenFaces[i][j] < min )
		                min = eigenFaces[i][j];
		        }
		        
		        double r=max-min;
		        for (int j = 0; j < hw; j++)
		        {
		        	eigenFacesScaled[i][j] = (byte) (((eigenFaces[i][j] - min)/r)*255);
		        }	        
	        }        
 
		    //Create objects from eigen faces
		    Utils.fillEigenImagesArray(eigenFacesScaled, eigenFacesArray);
	
		    //Array of ImageViews
		    for(int i = 0; i < numEFaces; i++)
		    {	
		    	ImageView im1 = new ImageView();//Create an ImageView
		    	im1.setImage(eigenFacesArray[i].img);//Set The Image
		    	imageViewArray[i] = im1;//Add to the Array
		    }
		    
		    //Display the Eigen Face Images
		    eFacesHBox.getChildren().addAll(imageViewArray);	    
		    trainingScrollPane.setContent(eFacesHBox); 
		    		   
		    //transpose the eigenFaces
		    double[][] eigenFacesTransposed = new double[hw][numEFaces];
		     for (int i = 0; i < numEFaces; i++) {
		            for (int j = 0; j < hw; j++) {
		            	//eigenFacesTransposed[j][i] = (double)((eigenFacesScaled[i][j] & 0xff));
		            	eigenFacesTransposed[j][i] = eigenFaces[i][j];
		            }
		        }
			eigenFacesTransposedGlobal = eigenFacesTransposed;
			
		    //Calculate the weights.
		    double[][] weightsForAssignment = new double[ti][numEFaces];
		    for(int i=0; i< ti; i++)
		    {
		        for(int j=0; j<numEFaces; j++)
		        {
		        	weightsForAssignment[i][j]=0;  
		          for(int k=0; k< hw; k++)
		          {
		        	  weightsForAssignment[i][j] += trainingImgsNormalized[i][k]*eigenFacesTransposed[k][j];
		          }
		        }
		    }
		    weights = weightsForAssignment;
		    

		    //VERIFIER
//		    int count = 0;
//		     for (int i = 0; i < ti; i++) {
//		            for (int j = 0; j < numEFaces; j++) {
//		            	//System.out.println(weightsForAssignment[i][j]);
//		            	count++;
//		            }
//		        }
//         	System.out.println(count);

		}
		else
		{
			System.out.println("Value must be less than total images and > 0");
		}
		}
		catch(Exception e)
		{
			System.out.println("Must Enter a Value for training Images");
		}
		

	}//End Training Image Button		
	
	@FXML
	protected void nextImageView(ActionEvent event)
	{	
		try
		{
		if(num < ti)
		{
			num++;
			Image oriImg = faces[num].img;
			updateImageView(oriImage, oriImg);
			oriImgNameLbl.setText(faces[num].name);
			oriImgNumLbl.setText(Integer.toString(faces[num].id));			
		}
		else if(num == ti-1)
		{
			Image oriImg = faces[ti-1].img;
			updateImageView(oriImage, oriImg);
			oriImgNameLbl.setText(faces[num].name);
			oriImgNumLbl.setText(Integer.toString(faces[num].id));
		}
		}
		catch(Exception e)
		{
			System.out.println("No Images Loaded");
		}
	}

	@FXML
	protected void prevImageView(ActionEvent event)
	{	
		try
		{
		if(num > 0)
		{
			num--;
			Image oriImg = faces[num].img;
			updateImageView(oriImage, oriImg);
			oriImgNameLbl.setText(faces[num].name);
			oriImgNumLbl.setText(Integer.toString(faces[num].id));
		}
		else if(num == 0)
		{
			Image oriImg = faces[0].img;
			updateImageView(oriImage, oriImg);
			oriImgNumLbl.setText(Integer.toString(faces[num].id));
		}
		}
		catch(Exception e)
		{
			System.out.println("No Images Loaded");
		}
	}
	
	@FXML
	protected void firstImageView(ActionEvent event)
	{	
		try
		{
			if(num >= 0)
			{
				num = 0;
				Image oriImg = faces[0].img;
				updateImageView(oriImage, oriImg);
				oriImgNameLbl.setText(faces[0].name);
				oriImgNumLbl.setText(Integer.toString(faces[0].id));
			}
			else
			{	
			}
		}
		catch(Exception e)
		{
			System.out.println("No Images Loaded");
			
		}
	}
	
	@FXML
	protected void lastImageView(ActionEvent event)
	{	
		try
		{
			if(num <= ti)
			{
				num = ti-1;
				Image oriImg = faces[ti-1].img;
				updateImageView(oriImage, oriImg);
				oriImgNameLbl.setText(faces[ti-1].name);
				oriImgNumLbl.setText(Integer.toString(faces[ti-1].id));				
			}
		}
		catch(Exception e)
		{
			System.out.println("No Images Loaded");
		}
	}

	//Get the Total count of files in the directory
	@FXML
	protected int tTabHandler(ActionEvent event)
	{		
		int ti = Utils.countFilesOnly(files,0);
		return ti;
	}
	
	//-------------------------------------------FACE RECOGNITION TAB-------------------------------------------------------------------//
	
	//Capture Image button on face recognition Tab
	@FXML
	protected void captureImageFaceRec(ActionEvent event)
	{
		//Set a default resize
		int h = 128;
		int w = 128;
		Mat ROI = new Mat();

		Mat frame = new Mat();
		// Now detect the face in the image.
	    // MatOfRect is a special container class for Rect.
		MatOfRect faceDetections = new MatOfRect();
	
		capture.retrieve(frame);
		
		faceDetectorClassifier.detectMultiScale(frame, faceDetections); //detectMultiScale will perform the detection
	    //Draw a bounding box around each face.
	    for (Rect rect : faceDetections.toArray()) {
		    Imgproc.rectangle(frame, 
		    new Point(rect.x, rect.y),   //bottom left
		    new Point(rect.x + rect.width, rect.y + rect.height), //top right 
		    new Scalar(0,255,0)); //RGB colour
		    
		    //This operation extracts the frame within the face recognition box
		    ROI = frame.submat(rect.y + 10, (rect.y + rect.height) - 10, rect.x + 10, (rect.x + rect.width) - 10); 
	     }
	    
	    //After the frame is cropped this function will resize the Image
	    Size sz = new Size(h,w);
	    Imgproc.resize(ROI,newROI,sz);
	    
		//capture a gray image
		Imgproc.cvtColor(newROI, newROI, Imgproc.COLOR_BGR2GRAY);//.COLOR_BGR2GRAY
		Image imageToShow = Utils.mat2Image(newROI);
		updateImageView(frameCapFaceRecTab, imageToShow);
		System.out.println("Width: " + imageToShow.getWidth() + " Length: " + imageToShow.getHeight());
	}	
	
	@FXML
	protected void recognizeBtnClick(ActionEvent event) throws IOException
	{
		try
		{
		Face newFace =  new Face();
		//Get image from the Image view
		Image img1 = frameCapFaceRecTab.getImage();	
		//------------------------------------------------------------------------------------------------------------//
		BufferedImage ImageToWrite = SwingFXUtils.fromFXImage(img1, null);
		saveImage(ImageToWrite,"D:\\Matthew\\workspace\\finalProject\\example.png");
    	newFace.bImg = ImageIo.readImage("D:\\Matthew\\workspace\\finalProject\\example.png");//Store bufferedImage
    	//newFace.bImg = Utils.convert4BYTETO3BYTE(newFace.bImg);
    	//newFace.img = SwingFXUtils.toFXImage(newFace.bImg, null);//Convert from buffered to Image
    	newFace.imageData = ImageIo.getDoubleImageArray1DFromBufferedImage(newFace.bImg);//1D Row Image 
        //------------------------------------------------------------------------------------------------------------//
		//TEST
//		newFace.bImg = ImageIo.readImage("example.png");
//		newFace.imageData = ImageIo.getDoubleImageArray1DFromBufferedImage(newFace.bImg);//1D Row Image 
		//-----------------------------------------------------------------------------------------------------------//
		int width = newFace.imageData.length;

		double [] originalImg = new double [width];
		originalImg = newFace.imageData;
	
		//Normalize the image
		double[] originalImgNormalized = new double[width];
	    double[] mvals = new double[originalImgNormalized.length];
	    
	    originalImgNormalized = originalImg;
	    
	    for (int i = 0; i < originalImgNormalized.length; i++)
	      mvals[i] = Utils.max(originalImgNormalized);

	    for (int i = 0; i < originalImgNormalized.length; i++) {
	    	  originalImgNormalized[i] /= mvals[i];
	    }//End Normalization 
	    		
		//Subtract the Normalized avg from the original normalized img
	    double[] newImgAvgNormalized = new double[width];
		for(int i = 0; i < width ;i++)
		{
			newImgAvgNormalized[i] = originalImgNormalized[i] - avgImageGlobal[i];
		}
		
	    //Calculate the weights of the new input image
	    double[] newImgWeight = new double[numEFacesGlobal];

	        for(int j=0; j<numEFacesGlobal; j++)
	        {
	        	newImgWeight[j]=0;  
	          for(int k=0; k< width; k++)
	          {
	        	  newImgWeight[j] += newImgAvgNormalized[k]*eigenFacesTransposedGlobal[k][j];
	          }
	        }
	    	
	    //Transpose the weights
	    double[][] weightsTransposed = new double[numEFacesGlobal][ti];
	     for (int i = 0; i < ti; i++) {
	            for (int j = 0; j < numEFacesGlobal; j++) {

	            	weightsTransposed[j][i] = weights[i][j];
	            }
	        }
	     
	    //Calculate the distances 
		double[] distances  = new double[ti];
	    for(int i=0; i< ti; i++)
	    {
          for(int j=0; j < numEFacesGlobal ; j++)
	          {
	        	  distances[i] += ((newImgWeight[j]-weightsTransposed[j][i])*(newImgWeight[j]-weightsTransposed[j][i]));
	          }
	    }
	    
	    //SquareRoot the distances
	    for(int i = 0; i < distances.length; i++)
	    {
	    	distances[i] = Math.sqrt(distances[i]);
	    }
	    
        double min = distances[0];
        int index = 0;
        for(int i = 0; i < distances.length; i++)
        {
            if(distances[i] < min)
            {
                min = distances[i];
                index = i;
            }
        }
	    
	    System.out.println("Smallest Distance is: " + min + ". Index is: " + index);
	    System.out.println("Person Identified as: " + faces[index].name);
	    
	    displayNameLbl.setText(faces[index].name);
	    
		Image recognizedImage = faces[index].img;
		updateImageView(finalFaceImgView, recognizedImage);    
	    
		
		
		//VERIFIER
//	    System.out.println("\nNEW IMAGE WEIGHTS");
//	    for(int i = 0; i < newImgWeight.length;i++)
//	    	System.out.println(newImgWeight[i]);
//	    System.out.println("\nTRAINING IMAGE WEIGHTS");
//	    
//	    for(int i = 0; i < weights.length;i++)
//	    	for(int j = 0; j < weights[0].length; j++)
//	    	{
//			    {
//			    	System.out.println(weights[i][j]);
//			    }
//	    	}
//	    
//	    System.out.println("\nDISTANCES: ");
//	    for(int i = 0; i < distances.length; i++)
//	    {
//	    	System.out.println(distances[i]);
//	    }
		}
		catch(Exception e)
		{
			System.out.println("Must Create Training First");
			//System.out.println(e);
		}
	
	}
	

 

}