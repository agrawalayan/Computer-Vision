###import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

###pre-processing of the image
image_object = cv2.imread("TestImages/test1.jpg")
original_image = image_object
gray_image = cv2.cvtColor(image_object, cv2.COLOR_BGR2GRAY)
gaussian_filtered_image = cv2.GaussianBlur(gray_image,(3,3),0)
ret_value,binary_image = cv2.threshold(gaussian_filtered_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("binary_image.jpg",binary_image)
print(binary_image.shape[:2])


###Horizonatl projection
image_2d_matrix = np.array(binary_image)
#print("MATRIX SIZE FOR LINE SEGMENTATION:", image_2d_matrix.shape)
horizontal_project_matrix = (image_2d_matrix!=255).sum(1)
line_segmentation = []
line_width_count = 0
for projection_pixels in range(len(horizontal_project_matrix)):
    if horizontal_project_matrix[projection_pixels] > int(image_object.shape[1]/50):   #if the line containe 2% of text
        line_width_count += 1
        if line_width_count == 1:
            start_line_index = projection_pixels
        continue
    if line_width_count >= 1:
        end_line_index = projection_pixels - 1
        if end_line_index - start_line_index > 30:                                     #if the text size is very small ~ close to not seen
            line_segmentation.append((start_line_index, end_line_index))
    line_width_count = 0

for lines in line_segmentation:
    cv2.rectangle(image_object, (0, lines[0]), (image_object.shape[1], lines[1]), (0, 255, 0), 2)

cv2.imwrite("line_segmentation.jpg",image_object)

DETECTED_WORD_BOUNDING_BOX = []
###Vertical Projection
for lines in line_segmentation:
    crop_lines_from_image = binary_image[lines[0]:lines[1], 0:image_object.shape[1]]
    cropped_line_2d_matrix = np.array(crop_lines_from_image)
    #print("MATRIX SIZE FOR WORD SEGMENTATION:", cropped_line_2d_matrix.shape)
    vertical_project_matrix = (cropped_line_2d_matrix!=255).sum(0)
    word_segmentation = []
    word_width_counting = 0
    for projection_pixels in range(len(vertical_project_matrix)):
        if vertical_project_matrix[projection_pixels] > 7:
            word_width_counting += 1
            if word_width_counting == 1:
                start_word_index = projection_pixels
            continue
        if word_width_counting >= 1:
            end_word_index = projection_pixels - 1
            if end_word_index - start_word_index > 15:
                word_segmentation.append((start_word_index, end_word_index))
                DETECTED_WORD_BOUNDING_BOX.append((start_word_index,lines[0],end_word_index,lines[1]))
            else:
                continue
        word_width_counting = 0
    #print(word_segmentation)

#print(DETECTED_WORD_BOUNDING_BOX)
for words in DETECTED_WORD_BOUNDING_BOX:
    cv2.rectangle(image_object, (words[0], words[1]), (words[2], words[3]), (0, 0, 255), 2)

cv2.imwrite("word_segmentation.jpg", image_object)


###Character Segmentation
##mser = cv2.MSER_create()
##original_image = cv2.resize(original_image, (original_image.shape[1], original_image.shape[0]))
##gray_image_1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
##regions = mser.detectRegions(gray_image_1)
##region_with_character = []
##character_index = 0
##for x,y,w,h in regions[1]:
##    if w < 40 and h < 40:
##        region_with_character.append(regions[0][character_index])
##        cv2.rectangle(image_object, (x, y), (x + w, y + h), (255, 0, 0), 2)
##    character_index += 1
##
##cv2.imwrite("character_segmentation.jpg",image_object)



