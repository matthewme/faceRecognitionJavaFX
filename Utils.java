package finalProject;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;

import org.opencv.core.Mat;

import cern.colt.list.DoubleArrayList;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.stat.Descriptive;
import finalProject.Face;
import finalProject.ImageIo;
import javafx.application.Platform;
import javafx.beans.property.ObjectProperty;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;

/**
 * Provide general purpose methods for handling OpenCV-JavaFX data conversion.
 * Moreover, expose some "low level" methods for matching few JavaFX behavior.
 *
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @author <a href="http://max-z.de">Maximilian Zuleger</a>
 * @version 1.0 (2016-09-17)
 * @since 1.0
 * 
 */
public final class Utils
{

	/**
	 * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
	 *
	 * @param frame
	 *            the {@link Mat} representing the current frame
	 * @return the {@link Image} to show
	 */
	public static Image mat2Image(Mat frame)
	{
		try
		{
			return SwingFXUtils.toFXImage(matToBufferedImage(frame), null);
		}
		catch (Exception e)
		{
			System.err.println("Cannot convert the Mat object: " + e);
			return null;
		}
	}
	
	/**
	 * Generic method for putting element running on a non-JavaFX thread on the
	 * JavaFX thread, to properly update the UI
	 * 
	 * @param property
	 *            a {@link ObjectProperty}
	 * @param value
	 *            the value to set for the given {@link ObjectProperty}
	 */
	public static <T> void onFXThread(final ObjectProperty<T> property, final T value)
	{
		Platform.runLater(() -> {
			property.set(value);
		});
	}
		
	/**
	 * Support for the {@link mat2image()} method
	 * 
	 * @param original
	 *            the {@link Mat} object in BGR or grayscale
	 * @return the corresponding {@link BufferedImage}
	 */
	private static BufferedImage matToBufferedImage(Mat original)
	{
		// init
		BufferedImage image = null;
		int width = original.width(), height = original.height(), channels = original.channels();
		byte[] sourcePixels = new byte[width * height * channels];
		original.get(0, 0, sourcePixels);
		
		if (original.channels() > 1)
		{
			image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
		}
		else
		{
			image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
		}
		final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.length);
		
		return image;
	}
	
	public static void showFiles(File[] files,int count, Face[] fArray, int num) {
    for (File file : files) {
        if (file.isDirectory()) {
            //System.out.println("Directory: " + file.getName());
            showFiles(file.listFiles(),num,fArray,num);
            num += file.listFiles().length;
        } else {
            //System.out.println("File: " + file.getName());
            System.out.println("The Count is: " + count);
			fArray[count] = new Face();//Create a face and store in the array of face objects
        	fArray[count].bImg = ImageIo.readImage(file.getPath());//Store bufferedImage
        	fArray[count].name = file.getName();
        	fArray[count].img = SwingFXUtils.toFXImage(fArray[count].bImg, null);//Convert from buffered to Image
        	fArray[count].imageData = ImageIo.getDoubleImageArray1DFromBufferedImage(fArray[count].bImg);//1D Row Image 
        	fArray[count].id = count;
        	count++;
        }
    }
}
	
	//Returns the total of all files in a directory and its subdirectories
	public static int countFilesOnly(File[] files, int num) 
	{
	    for (File file : files) {
	        if (file.isDirectory()) {
	            //System.out.println("Directory: " + file.getName());
	            countFilesOnly(file.listFiles(),num);
	            num += file.listFiles().length;
	        }
	    }
	    return num;
    }
	
	public static void fillEigenImagesArray(byte[][] eigenFacesScaled, Face[] fArray)
	{
	    
//	    byte[][] eFace = new byte[(int)Math.sqrt(eigenFacesScaled[0].length)][(int)Math.sqrt(eigenFacesScaled[0].length)];
	    for(int i=0; i < eigenFacesScaled.length; i++)
	    {
	    	byte[][] eFace = new byte[(int)Math.sqrt(eigenFacesScaled[0].length)][(int)Math.sqrt(eigenFacesScaled[0].length)];
	        for(int j=0; j<eigenFacesScaled[0].length; j++)
	        {
	        	//Convert each row into a 2-d Array
	        	int row = j/(int)Math.sqrt(eigenFacesScaled[0].length);
		    	int col = j%(int)Math.sqrt(eigenFacesScaled[0].length);
		    	eFace[row][col] = eigenFacesScaled[i][j]; 
	        }
	        
	        //Pass the 2D byte image data to the object
	        fArray[i] = new Face();//Create a face and store in the array of face objects
	        fArray[i].eigenFaceByteData = eFace;
	        //Turn it into an Image and assign to object
	        BufferedImage img = ImageIo.setGrayByteImageArray2DToBufferedImage(eFace);
	        fArray[i].img = SwingFXUtils.toFXImage(img, null);        
	    }
	}
	
	
	public static DoubleMatrix2D calcCovarMat(double[][] trainingData)
	{ 
		int numRows = trainingData.length;
		DoubleMatrix2D matCov = new DenseDoubleMatrix2D(numRows, numRows); 
		for (int i = 0; i < numRows; i++)
		{ 
			DoubleArrayList iRow = new DoubleArrayList(trainingData[i]); 
			double variance = Descriptive.covariance(iRow, iRow); 
			matCov.setQuick(i, i, variance); // main diagonal value 
			// fill values symmetrically around main diagonal 
			for (int j = i+1; j < numRows; j++) 
			{ 
				double cov = Descriptive.covariance(iRow, new DoubleArrayList(trainingData[j]));
				matCov.setQuick(i, j, cov); // fill to the right
				matCov.setQuick(j, i, cov); // fill below 
			} 
		}
		return matCov; 
	}
	
	///-------------------------Begin sort eigens------------------------------------//
	  public static void sortEigenInfo(double[] egVals, double[][] egVecs)
	  /* sort the Eigenvalues and Eigenvectors arrays into descending order
	     by eigenvalue. Add them to a table so the sorting of the values adjusts the
	     corresponding vectors
	  */
	  {
	    Double[] egDvals = getEgValsAsDoubles(egVals);

	    // create table whose key == eigenvalue; value == eigenvector
	    Hashtable<Double, double[]> table = new Hashtable<Double, double[]>();
	    for (int i = 0; i < egDvals.length; i++)
	      table.put( egDvals[i], getColumn(egVecs, i) );     

	    ArrayList<Double> sortedKeyList = sortKeysDescending(table);
	    updateEgVecs(egVecs, table, egDvals, sortedKeyList);
	       // use the sorted key list to update the Eigenvectors array

	    // convert the sorted key list into an array
	    Double[] sortedKeys = new Double[sortedKeyList.size()];
	    sortedKeyList.toArray(sortedKeys); 

	    // use the sorted keys array to update the Eigenvalues array
	    for (int i = 0; i < sortedKeys.length; i++)
	      egVals[i] = sortedKeys[i].doubleValue();

	  }  // end of sortEigenInfo()
	  
	  private static void updateEgVecs(double[][] egVecs, Hashtable<Double, double[]> table, Double[] egDvals, ArrayList<Double> sortedKeyList)
		/* get vectors from the table in descending order of sorted key,
		and update the original vectors array */
		{ 
		for (int col = 0; col < egDvals.length; col++) {
		double[] egVec = table.get(sortedKeyList.get(col));
		for (int row = 0; row < egVec.length; row++) 
		egVecs[row][col] = egVec[row];
		}
		}  // end of updateEgVecs()

	  private static Double[] getEgValsAsDoubles(double[] egVals)
	  // convert double Eigenvalues to Double objects, suitable for Hashtable keys
	  {  
	    Double[] egDvals = new Double[egVals.length];
	    for (int i = 0; i < egVals.length; i++)
	      egDvals[i] = new Double(egVals[i]);
	    return egDvals;
	  }  // end of getEgValsAsDoubles()
	  
	  private static double[] getColumn(double[][] vecs, int col)
	  /* the Eigenvectors array is in column order (one vector per column);
	     return the vector in column col */
	  {
	    double[] res = new double[vecs.length];
	    for (int i = 0; i < vecs.length; i++)
	      res[i] = vecs[i][col];
	    return res;
	  }  // end of getColumn()

	  private static ArrayList<Double> sortKeysDescending(
	                                        Hashtable<Double,double[]> table)
	  // sort the keylist part of the hashtable into descending order
	  {
	    ArrayList<Double> keyList = Collections.list( table.keys() );
	    Collections.sort(keyList, Collections.reverseOrder()); // largest first
	    return keyList;
	  }  // end of sortKeysDescending()
	 //--------------------------------------end sort eigens-----------------------------//
	  
	  static double max(double[] arr)
	  {
	    double max = Double.MIN_VALUE;
	    for (int i = 0; i < arr.length; i++)
	      max = Math.max(max, arr[i]);			
	    return max;
	  }  // end of max()
	  
	  //Convert from TYPE__4BYTE_ABGR to TYPE_3BYTE_BGR
	  public static BufferedImage convert4BYTETO3BYTE(BufferedImage bufferedImage)
	  {
		  if (bufferedImage.getType() == BufferedImage.TYPE_4BYTE_ABGR) {
		      BufferedImage bff = new BufferedImage(bufferedImage.getWidth(), bufferedImage.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
		      for (int y = 0; y < bufferedImage.getHeight(); ++y) {
		          for (int x = 0; x < bufferedImage.getWidth(); ++x) {
		          int argb = bufferedImage.getRGB(x, y);
		          bff.setRGB(x, y, argb & 0xFF000000); // same color alpha 100%
		          }
		      }
		      return bff;
		  } else {
		      return bufferedImage;
		  }
	  }
	 
}