import cv2
import numpy as np

def unsharpFuc(image, k=2.0):

    # get the images dimensions to be able to make it unsharp
    height, width = image.shape

    # apply padding to the image using the 'constant' method
    paddedImage = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT)

    # create a 3x3 averaging mask
    avgMask = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]]) / 9.0

    # put the image together with the mask
    # make sure that we ads the passing which we calculated from the top
    blurredImage = cv2.filter2D(paddedImage, -1, avgMask)

    # calculate the unsharp mask by subtracking the blured
    #  image from the orig image to make sure we do not include the pixels in the border

    unsharp = cv2.subtract(image, blurredImage[1:height+1, 1:width+1])

    # add the unsharp mask back to the original image
    resultImage = cv2.add(image, k * unsharp, dtype=cv2.CV_8U)  # specify the data type

    return resultImage

def main():
    # ask the user for an image file name
    imageFile = input("enter the image file name: ")

    # load the image as a grayscale image
    originalImage = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)

    if originalImage is None:
        print("eror: unable to open the image.")
        return

    # perform unsharp masking with an increased 'k' value for a more dramatic effect
    resultImage = unsharpFuc(originalImage, k=2.0)

    # save the this as a new image
    outputFile = "unsharp_" + imageFile
    cv2.imwrite(outputFile, resultImage)

    # display the original and the updates image with edits
    cv2.imshow("original image", originalImage)
    cv2.imshow("unsharp masked image", resultImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
