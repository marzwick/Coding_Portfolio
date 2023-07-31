# This is a prgoram for analyzing clustering of magnetic nanoparticles and bonding between magnetic nanoparticles and adeno-associated viruses in transmission electron microscopy (TEM) images

# import statements
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics

# helper functions
def create_df():
    #creates a dataframe for mnp's with the aav's close to a mnp and whether it is in a cluster
    columns = ["aav_list", "cluster?", "not_on_border?"]
    df = pd.DataFrame(data=None, index=None, columns=columns, dtype=None, copy=None)
    return df

def jpg_num(image):
    #removes the .jpg from the image title
    num = image[:-4]
    return num

def show_img(array):
    #displays a numpy array
    plt.imshow(array)
    plt.axis('off')
    plt.show()
    
def find_coords(contour):
    #finds the x and y coordinates of the centroid of a contour
    moments = cv2.moments(contour)
    if moments["m00"] != 0: #checks for non-zero area
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        return (0,0) #for incorrect particles
    return cx,cy

def find_centroid_list(count_list):
    #returns a list of tuples containing (x,y) coordinates of contour centroid from the input arrays
    centroids_list = []
    for contour in count_list:
        centroids_list.append(find_coords(contour))
    return centroids_list

def find_key(dictionary, item):
    #finds the key of the list containing the item
    for key, value in dictionary.items():
        if item in value:
            return key
    return None

def check_elements_contained(small_list, large_list):
    #checks if all elements in the small list are in the large list
    return all(element in large_list for element in small_list)

def check_elements_in_dict_values(dictionary, new_list):
    for sublist in dictionary.values():
        if check_elements_contained(new_list, sublist):
            return True
    return False

def find_distance(mnp, aav):
    #finds the distance between two particles
    return math.sqrt((mnp[0] - aav[0])**2 + (mnp[1] - aav[1])**2)

def get_circularity(contour):
    # Calculate the area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate circularity
    if perimeter!=0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        return circularity
    return 1

def get_aspect_ratio(contour):
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    try:
        aspect_ratio = max(width, height) / min(width, height)    
        return aspect_ratio
    except ZeroDivisionError:
        return 1

def get_median_area(count_list):
    area_list=[]
    for contour in count_list:
        area_list.append(cv2.contourArea(contour))
    return statistics.median(area_list)

def particle_counts(sample_img, mnptf, test):
    #finds the number of mnps or aavs and their centroids, and creates an image with contours drawn over them
    
    num = jpg_num(sample_img) #finds the name of the image
    
    gray = cv2.imread(sample_img, 0) #converts to numpy grayscale array
    
    if mnptf:
        mv = np.median(gray) #finds the median pixel value
    
    blur = cv2.medianBlur(gray, 3) #median blur (reduce noise)
    blur = cv2.GaussianBlur(blur, (5,5), 0) #gaussian blur (reduce noise)
    
    if mnptf: #applies the mnp contrast and canny
        contrast = cv2.convertScaleAbs(blur, alpha=1.925, beta=3.5) #creates a contrast, alpha scales and beta adds to each
        canny = cv2.Canny(contrast, 1, 8) #determines edge pixels with 60 as the lower threshold and 150 as the higher threshold
        
    else: #applies the aav contrast and canny
        contrast = cv2.convertScaleAbs(blur, alpha=1.435, beta=0.5)
        canny = cv2.Canny(contrast, 6, 11)
    if mnptf:
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)) #dilates image after contrasting
    else:
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)) #dilates image after contrasting
    dilated = cv2.dilate(canny, kernel_dilation, iterations=1)
    
    (cnt, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #finds contours after image has been thresholded
    rgb = cv2.imread(sample_img)  # Read the original image in BGR format

    if mnptf:
        cv2.drawContours(rgb, cnt, -1, (mv, mv, mv), -1) # Fill outside contours for mnps to ensure everything is filled
        cv2.drawContours(rgb, cnt, -1, (mv, mv, mv), 8) # Fill outside contours for mnps to ensure everything is filled
        cv2.imwrite(f'{num}_mnp.jpg', rgb) #saves the image with the contours for mnp
    elif test:
        cv2.drawContours(rgb, cnt, -1, (0,255,0), -1) # Fill contours with median color for mnps, green for aavs
        cv2.imwrite(f'{num}_aav.jpg', rgb) #saves the image with the contours for aav
    particle_count_list = cnt #gives data on each particle
    
    return particle_count_list

def is_contour_on_border(image, contour, thickness):
    height, width = image.shape[:2]

    # Iterate over each point in the contour
    for point in contour:
        x, y = point[0]

        # Check if the point is outside the image boundaries
        if x < thickness or x >= (width - thickness) or y < thickness or y >= (height-thickness):
            return True  # Contour extends beyond the border
    return False

def corresponding(count_list1, count_list2, df, aavtf, test, inversetf):
    corresponding_dict={} #dict with indexes in count_list1 as keys and lists of particles in count_list2 close to each index in list1 as the values
            
    image1 = np.zeros((668, 1002), dtype=np.uint8) #creates two blank images to check for overlap
    image2 = np.zeros((668, 1002), dtype=np.uint8) #second image with contours from count_list2
    
    for index, contour in enumerate(count_list1):
        
        touching_list=[] #list of particles in count_list2 touching the particle at index in count_list1
        
        coords1=find_coords(contour)
        
        if inversetf: #if inverted, then count_list1 is aavs and count_list2 is mnps
            thickness=4
        else:
            thickness=8
            
            
        if not(aavtf or inversetf): # checks that both count lists are mnps
            if is_contour_on_border(image1, contour, 5):
                df.at[index, "not_on_border?"] = False # the mnp is on the border
        
        cv2.drawContours(image1, [contour], -1, 255, thickness) # Draw contour1 on image1
                
        for index2, contour2 in enumerate(count_list2): # iterates through count_list2
            
            coords2=find_coords(contour2) # coordinates of second particle
            
            if find_distance(coords1, coords2)<(6*(math.sqrt(cv2.contourArea(contour)/3.141)+math.sqrt(cv2.contourArea(contour2)/3.141))): # easy check to see if they're really far away

                if aavtf: #different thickness around aavs then mnps
                    thickness2=4
                else:
                    thickness2=8
                cv2.drawContours(image2, [contour2], -1, 255, thickness2) # Draw contour2 on image2


                intersection = cv2.bitwise_and(image1, image2) # Find intersection between contours

                if cv2.countNonZero(intersection) > 0: # Check if there is any intersection
                    touching_list.append(index2)

                cv2.drawContours(image2, [contour2], -1, 0, thickness2) # Erase contours on image2
            
        if not(aavtf or inversetf or df.at[index, "not_on_border?"]): #if it's an mnp on the border, all mnps it touches are also on the border
            for index2 in touching_list:
                df.at[index2, "not_on_border?"] = False #the mnp is on the border
                
        corresponding_dict[index]=touching_list
        cv2.drawContours(image1, [contour], -1, 0, thickness) # Erase contours on image1
        
    if aavtf: #mutates the dataframe to add the aav lists
        for mnp, aavs in corresponding_dict.items():
            df.at[mnp, "aav_list"] = aavs
        if test:
            print("AAV Dict:")
            print(corresponding_dict)  
            print()
            
    if not(aavtf or inversetf):
        for index, row in df.iterrows():
            if row['not_on_border?'] != False:
                df.at[index, 'not_on_border?'] = True
    
    return corresponding_dict

def merged(corresponding_dict, test):
    #returns a dictionary with indexes as keys and lists of indexes of mnps in a cluster with the key mnp as the values
    num_cluster=0 #number of total clusters
    checked=set() #indexes that have been checked already
    merged_dict={} #dictionary to be returned
    for key, value in corresponding_dict.items():
        if key not in checked: #don't want to recheck a key
            cluster_list=find_cluster(corresponding_dict,key, set()) #recursive function that finds the indexes of mnps in the cluster
            checked.update(cluster_list) #add these new indexes in the cluster list to checked so they won't be checked later
            merged_dict[key]=sorted(cluster_list) #adds the cluster list to the dictionary

    if test:
        print("Clusters pre update:")
        print(merged_dict)
    return merged_dict
                
    
def find_cluster(corresponding_dict,key, set_cluster):
    #used with merged to find the list of indexes in a cluster 
    to_check=[]
    list_indexes=corresponding_dict[key]
    for index in list_indexes:
        if index not in set_cluster and index!=key: #checks for new indexes that needs to have their value lists checked
            to_check.append(index)
    set_cluster.update(list_indexes) #adds new indexes in the value list for the key
    if len(to_check)!=0: #in order to stop recursion
        for index in to_check:
            set_cluster.update(find_cluster(corresponding_dict,index, set_cluster)) #recursion to check any new indexes not in the set cluster
    return list(set_cluster)

def update_merged(merged_dict, mnp_corres_aav,test, df):
    
    for mnp_grouping in mnp_corres_aav.values(): #finds the mnps around an aav to put them in a group
        if not check_elements_in_dict_values(merged_dict, mnp_grouping): #checks that it is by multiple mnps and that this is not already a cluster
            key_list=[]
            for mnp in mnp_grouping:
                if mnp in merged_dict.keys():
                    key_list.append(mnp)
                else:
                    key_list.append(find_key(merged_dict, mnp))
            mnp_list_total=[]
            for key in key_list:
                mnp_list_total.extend(merged_dict[key])
            for key in key_list[1:]:
                if key in merged_dict.keys():
                    del merged_dict[key]
            merged_dict[key_list[0]]=list(set(mnp_list_total))
                
    num_cluster=0
    for mnps in merged_dict.values():
        if len(mnps) > 2:
            df.loc[mnps, "cluster?"] = True
            if True: #check_boolean(df, "not_on_border?", mnps):
                num_cluster+=1
        else:
            df.loc[mnps, "cluster?"] = False
                    
    if test:
        print("MNP Dict:")
        print(merged_dict)
    return num_cluster, merged_dict
                
    
def check_boolean(df, boolean_column, index_list):
    # Get the boolean values for the given indexes
    boolean_values = df.loc[index_list, boolean_column]

    # Check if the boolean values are True for all the indexes
    return boolean_values.all()

def results(df, clusters, mnp_corres_aav, aspect_ratio_dict, mnp_area, aav_count_list):
    #returns the result list

    result_list = []
    result_list.extend('c'*clusters) #adds cs to the result list equal to the number of clusters

    aav_dict={}
    result_list_small=[]
            
    for key, value in mnp_corres_aav.items():
        if (aspect_ratio_dict[key]>2.25 and cv2.contourArea(aav_count_list[key])>mnp_area*0.4) or cv2.contourArea(aav_count_list[key])>mnp_area:
            multiplier=2
        else:
            multiplier=1
        if value:
            aav_dict[key]=multiplier/len(value)
        else:
            aav_dict[key]=0
            
    for index, row in df.iterrows():
        if not(row["cluster?"] or not row["not_on_border?"]):
            counter=0
            aav_list=row["aav_list"]
            for num in aav_list:
                counter += aav_dict[num] #adds 1 if the aav is individual or 0.5 if it is shared
            result_list_small.append(counter)  
    result_list.extend(sorted(result_list_small, reverse=True))
    return result_list

def run_cell_count(img, test):
    #returns the result list when given an image
    
    num = jpg_num(img) #finds the name of the image
    image_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE) #loads the image
    median_value_img = np.median(image_array) #finds the median grayscale value of the image
    multiplier = 195/median_value_img #finds the multiplier to adjust the image
    scaled_image = (image_array * multiplier).astype(np.uint8) #adjusts the image
    cv2.imwrite(f'{num}_scaled.jpg', scaled_image) #saves it again under the name of the original image
    
    df = create_df() #creates a dataframe to track aav lists and clustering for each particle
    
    mnp_count_list = particle_counts(f"{jpg_num(img)}_scaled.jpg", True, test) #gets the mnp_count_list by performing operations on the image
    aav_count_list = particle_counts(f"{jpg_num(img)}_scaled_mnp.jpg", False, test) #gets the aav_count_list by performing operations on the mnp image
    corresponding(mnp_count_list, aav_count_list, df, True, test, False) #finds the neighboring aavs for each mnp
    mnp_groups = corresponding(mnp_count_list, mnp_count_list, df, False, test, False) #finds the neighboring mnps for each mnp
    mnp_corres_aav = corresponding(aav_count_list, mnp_count_list, df, False, test, True) #finds the neighboring mnps for each aav      
    aspect_ratio_dict=get_aspect_ratio_dict(aav_count_list)
    merged_groups = merged(mnp_groups, test) #finds the mnp clusters
    clusters, merged_groups = update_merged(merged_groups, mnp_corres_aav, test, df)
    filtered_df = df[(df['cluster?'] == False) & (df['not_on_border?'] == True)]
    mnp_area=get_median_area(mnp_count_list)
    result_list = results(filtered_df, clusters, mnp_corres_aav, aspect_ratio_dict, mnp_area, aav_count_list)
    
    if test:
        print(f"mnp_corres_aav:{mnp_corres_aav}")
        print()
        put_text(num, scaled_image, mnp_count_list, 0, False)
        put_text(num, scaled_image, aav_count_list, 255, True)
        print(df)
        print(filtered_df)
    return result_list

def put_text(num, scaled_image, count_list, color, aavtf):
    centroid_list=find_centroid_list(count_list)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    for index, centroid in enumerate(centroid_list):
        text = str(index)
        x = centroid[0]  # X-coordinate
        y = centroid[1]  # Y-coordinate

        cv2.putText(scaled_image, text, (x, y), font, font_scale, color, thickness=2) # Write the text on the image
    if aavtf:
        cv2.imwrite(f'{num}_aav_scaled_labeled.jpg', scaled_image) #saves it again under the name of the original image with labels
    else:
        cv2.imwrite(f'{num}_scaled_labeled.jpg', scaled_image) #saves it again under the name of the original image with labels
        
def get_aspect_ratio_dict(contours):
    aspect_ratio_dict={}
    for index, contour in enumerate(contours):
        aspect_ratio=get_aspect_ratio(contour)
        aspect_ratio_dict[index]=aspect_ratio
    return aspect_ratio_dict

 def analyze_images(start, stop, test):   
    entries=0
    num_0=0
    num_c=0
    total=0
    num_half=0
    
    for i in range(start,stop+1):
        sample_img=f"{i}.jpg"
        try:

            list_ele=run_cell_count(sample_img, test)
            num_0+=list_ele.count(0)
            num_c+=list_ele.count('c')
            num_half+=list_ele.count(0.5)

            numeric_list = [x for x in list_ele if isinstance(x, (int, float))]
            total+=sum(numeric_list)
            entries+=len(list_ele)
            print(f"{i}: {list_ele}")
            if test:
                print(list_ele.count(0))
                print(list_ele.count('c'))
        except FileNotFoundError:
            pass
        except TypeError:
            pass
    pct_0 = num_0/entries*100
    pct_c = num_c/entries*100
    avg = total/(entries-num_c)
    avg_no_0 = total/(entries-num_c-num_0)

    print(f"Percent 0: {pct_0}")
    print(f"Percent cluster: {pct_c}")
    print(f"Average aavs: {avg}")
    print(f"Average aavs no zeros: {avg_no_0}")