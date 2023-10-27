
#the rquired importss
import numpy as np
import cv2



def histogrammEqualization(img):
    # calculatte histogram of input image
    hist, _ = np.histogram(img, bins=256, range=(0, 256))

    # we have get the cummulatve dist of the 
    cdf = hist.cumsum()

    # nomalizze the cdf and convert it to the type to display the image
    maximmumCDF = cdf[-1]
    normallCDF = (cdf / maximmumCDF) * 255
    cdfNormal = normallCDF.astype(np.uint8)

    # apply the maping using the cdf
    outputImag = cdfNormal[img]

    return outputImag

def main():
    # get image file name from terminal from user
    imageFile = input("Enter the image file name: ")

    # show the image in grayscale
    orgImage = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)

    if orgImage is None:
        print("eror: unable to open the image.")
        return

    # rrun histogram equalisation in the function
    newImage = histogrammEqualization(orgImage)

    # display the original and resultng images
    cv2.imshow("original image", orgImage)
    cv2.imshow("equalised image", newImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save the result as a new image
    output_file = "equalized_" + imageFile
    cv2.imwrite(output_file, newImage)

if __name__ == "__main__":
    main()
