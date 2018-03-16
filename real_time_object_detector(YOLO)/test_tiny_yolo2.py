import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.misc as misc
import json, os, time, copy
import argparse
from datetime import datetime

SEG_SIZE = 7
IMAGE_SIZE = SEG_SIZE*64
CELL_SIZE = int(IMAGE_SIZE/SEG_SIZE)

COCO_ANNOTATIONS_PATH = '/home/minsung/COCO_DATASET/annotations/instances_train2014.json'
COCO_IMAGES_PATH = '/home/minsung/COCO_DATASET/train2014/'

def build_parser():
    parser = argparse.ArgumentParser()
    
        
    parser.add_argument('--model-load-path', type=str,
                        help='model load path',default='yolo_test_result2/yolo_tiny_model.ckpt')
    
    parser.add_argument('--gpu-num', type=str,
                        help='number of gpu to use', required=True)

    parser.add_argument('--test-result-path', type=str,
                        help='image result during training',default='/home/minsung/yolo_test_result/')
    
    parser.add_argument('--test-image-path', type=str,
                        help='test images file path', default='/home/minsung/image_for_yolo_test/')

    parser.add_argument('--threshold1', type=float,
                        help='confidence score threshold for detection procedure', default=0.1)
    parser.add_argument('--threshold2', type=float,
                        help='iou threshold for detection procedure', default=0.3)
    parser.add_argument('--threshold3', type=float,
                        help='result confidence threshold for detection procedure', default=0.0)
    return parser

##
json_file = json.load(open(COCO_ANNOTATIONS_PATH,'r'))
coco_annotations = json_file['annotations']
coco_images_info = json_file['images']
coco_categories = json_file['categories']

supercategoryName2categoryID = dict()
supercategoryID2name = dict()
supercategoryID2categoryID = dict()

ind = 0 
for category in coco_categories:
    category_id = category['id']
    category_name = category['name']
    supercategory_name = category['supercategory']
    if supercategory_name in supercategoryName2categoryID:
        supercategoryName2categoryID[supercategory_name].append(category_id)
    else:
        supercategoryName2categoryID[supercategory_name] = list([category_id])
        supercategoryID2name[ind] = supercategory_name
        ind += 1

for k,v in supercategoryID2name.items():
    supercategoryID2categoryID[k] = supercategoryName2categoryID[v]

del(json_file)
##
    


def plot_img_bbox(img, bbox,img_name):
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(img)
    for box in bbox:
        cenx,ceny,bw,bh = box[0:4]
        # minx = int(cenx - bw/2)
        # miny = int(ceny - bh/2)
        minx = cenx
        miny = ceny
        sup_id = box[-1]
        rect = patches.Rectangle((minx,miny), bw,bh, linewidth=1, edgecolor='r',facecolor='none')
        ob_name = supercategoryID2name[sup_id]
        plt.text(minx+bw/2, miny, ob_name,fontsize=20, color='green')
        ax.add_patch(rect)
    # plt.show()
    fig.savefig(img_name)

def visualize_img(x):
    return np.clip(x, 0, 255).astype('uint8')

def read_img(img_path,IMAGE_SIZE=IMAGE_SIZE):
    img = misc.imread(img_path)
    # if image is 1 channel stack 3 ch
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    
    return misc.imresize(img, (IMAGE_SIZE,IMAGE_SIZE,3))

def find_supercategory_id(category_id):
    for k,v in supercategoryID2categoryID.items():
        if category_id in v:
            return k

def get_coco_data(img_info,img_size=IMAGE_SIZE):
    """ 
        img_info : coco annotation's image information data
        return : image, bounding_boxes[object_num, x,y,w,h, super-category id]
    """
    img_file_name = img_info['file_name']
    img = read_img(COCO_IMAGES_PATH+img_file_name, img_size)
    img_h, img_w = img_info['height'], img_info['width']
    img_id = img_info['id']
    
    bounding_boxes = list() # x,y,w,h, super category id
    for annotation in coco_annotations:
        if annotation['image_id'] == img_id:
            category_id = annotation['category_id']
            super_id = find_supercategory_id(category_id)
            bbox = annotation['bbox']
            # change corresponding box's coord by IMAGE SIZE
            bbox= [bbox[0]/img_w, bbox[1]/img_h, bbox[2]/img_w, bbox[3]/img_h]
            bbox = [x*img_size for x in bbox]
            bounding_boxes.append(bbox+[1,super_id])
    return np.expand_dims(img,axis=0), np.array(bounding_boxes,dtype=np.float32)


def np_one_hot(x, depth=12):
    temp =[0]*12
    temp[int(x)]=1
    return temp


def assign_bbox(gboxes):
    """
        ground boxes shape chnage [?x5] -->[7,7,22]
        Assign a bounding box to the corresponding cell
        and box's x,y,w,h normalize to 0~1
    """
    cell_box = np.zeros([7,7,22], dtype=np.float32)
    obj_I = np.zeros([7,7,1], dtype=np.float32)
    no_I = np.ones([7,7,1], dtype=np.float32)
    for gbox in gboxes:
        box_minx, box_miny, box_w, box_h,box_conf, box_cate = gbox
        center_x = box_minx + box_w/2
        center_y = box_miny + box_h/2
        cell_x = int(center_x // CELL_SIZE)
        cell_y = int(center_y // CELL_SIZE)
        normalize_box = [ (center_x - cell_x*CELL_SIZE)/CELL_SIZE, (center_y-cell_y*CELL_SIZE)/CELL_SIZE, box_w/IMAGE_SIZE, box_h/IMAGE_SIZE, box_conf] 
        temp = normalize_box + normalize_box+ np_one_hot(box_cate)        
        cell_box[cell_y,cell_x,:] = temp
        obj_I[cell_y, cell_x,:] = 1
        no_I[cell_y, cell_x,:] = 0
        
    return cell_box,obj_I, no_I


def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def conv2d(h, w, stride=1):
    return tf.nn.conv2d(h, w, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(h, k=2, s=2):
    return tf.nn.max_pool(h, ksize=[1,k,k,1], strides=[1,s,s,1], padding="SAME")


def vgg16(h, train=True):
    pooling_architecture = [x*2 for x in [2, 4, 7, 9, 11, 13]]
    layer_i = 0
    pool_i = 0
    vggind = 0
    vgg16_weights = np.load('vgg16_weights.npy')
    while vggind < len(vgg16_weights):
        with tf.variable_scope('vgg16_layer'+str(layer_i)):
            w = tf.get_variable('conv_w'+str(vggind), initializer=vgg16_weights[vggind], trainable=train)
            vggind += 1
            b = tf.get_variable('conv_b'+str(vggind), initializer= vgg16_weights[vggind], trainable=train)
            vggind += 1
            h = lrelu(conv2d(h, w)+b)
            layer_i += 1
            if vggind == pooling_architecture[pool_i]:
                h = max_pool_2x2(h)
                pool_i += 1
    del(vgg16_weights)
    return h

def vgg19(h, train=True):
    pooling_architecture = [x*2 for x in [2, 4, 8,  11, 14, 16]]
    layer_i = 0
    pool_i = 0
    vggind = 0
    vgg19_weights = np.load('vgg19_weights.npy')
    while vggind < len(vgg19_weights):
        with tf.variable_scope('vgg19_layer'+str(layer_i)):
            w = tf.get_variable('conv_w'+str(vggind), initializer=vgg19_weights[vggind], trainable=train)
            vggind += 1
            b = tf.get_variable('conv_b'+str(vggind), initializer= vgg19_weights[vggind], trainable=train)
            vggind += 1
            h = lrelu(conv2d(h, w)+b)
            layer_i += 1
            if vggind == pooling_architecture[pool_i]:
                h = max_pool_2x2(h)
                pool_i += 1
    del(vgg19_weights)
    return h


def yolo_net(h):
    # preprocess
    h = tf.stack([ h[:,:,:,0] - 103.939, h[:,:,:,1] - 116.779,h[:,:,:,2] - 123.68 ], axis=3)
    layer_i = 0
    h = vgg16(h)

    # # # conv --> lrelu --> pooling layers
    # net_info = [16,32,64,128,256,512]
    # for filter_num in net_info:
    #     with tf.variable_scope('layer'+str(layer_i)):
    #         in_ch = h.get_shape().as_list()[-1]
    #         w = tf.get_variable('conv_w',[3,3,in_ch, filter_num],dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.05))
    #         b = tf.get_variable('conv_b', [filter_num], dtype=tf.float32, initializer=tf.zeros_initializer())
    #         h = lrelu(conv2d(h,w)+b)
    #         h = max_pool_2x2(h)
    #         layer_i += 1


    # conv --> lrelu layers
    net_info = [1024,1024,1024]
    for filter_num in net_info:
        with tf.variable_scope('layer'+str(layer_i)):
            in_ch = h.get_shape().as_list()[-1]
            w = tf.get_variable('conv_w',[3,3,in_ch, filter_num],dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.05))
            b = tf.get_variable('conv_b', [filter_num], dtype=tf.float32, initializer=tf.zeros_initializer())
            h = lrelu(conv2d(h,w)+b)
            layer_i += 1
    
    # flatten 3D tensor
    tensor_b, tensor_w,tensor_h, tensor_ch = h.get_shape().as_list()
    tensor_b = tf.shape(h)[0]
    h = tf.reshape(h, shape=[tensor_b, tensor_h*tensor_w*tensor_ch])
    
    # fully connected layer
    net_info = [4096]
    for linear_size in net_info:
        with tf.variable_scope('layer'+str(layer_i)):
            flattened_size = h.get_shape().as_list()[-1]
            w = tf.get_variable('linear_w', [flattened_size, linear_size], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.05))
            b = tf.get_variable('linear_b', [linear_size], dtype=tf.float32, initializer=tf.zeros_initializer())
            h = lrelu(tf.matmul(h,w)+b)
            layer_i += 1
    
    # output layer not use non-linear activation function
    output_size = SEG_SIZE*SEG_SIZE*(12+5*2)
    with tf.variable_scope('layer'+str(layer_i)):
        flattened_size = h.get_shape().as_list()[-1]
        w = tf.get_variable('linear_w', [flattened_size, output_size], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.05))
        b = tf.get_variable('linear_b', [output_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        h = tf.matmul(h,w)+b
        layer_i += 1
    
    
    # change flattend shape to 3D shape
    h = tf.reshape(h,[tensor_b, SEG_SIZE, SEG_SIZE, (12+5*2)])
    return h

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return (x,y,0,0)
    return (x, y, w, h)

def calc_iou(a,b):
    intersec = intersection(a,b)
    box1_size = a[2]*a[3]
    box2_size = b[2]*b[3]
    intersection_size = intersec[2]*intersec[3]
    return intersection_size / (box1_size + box2_size - intersection_size)

# def add_offset(bbox,cell_id):
#     box_x, box_y, box_w, box_h = bbox
#     box_w = box_w*IMAGE_SIZE
#     box_h = box_h*IMAGE_SIZE
#     x_offset = cell_id%SEG_SIZE
#     y_offset = cell_id//SEG_SIZE
#     return [(box_x+x_offset)*CELL_SIZE, (box_y+y_offset)*CELL_SIZE, box_w, box_h]

def add_offset(bbox,cell_id):
    box_cenx, box_ceny, box_w, box_h = bbox
    box_w = box_w*IMAGE_SIZE
    box_h = box_h*IMAGE_SIZE
    
    x_offset = (cell_id%SEG_SIZE)*CELL_SIZE
    y_offset = (cell_id//SEG_SIZE)*CELL_SIZE
    box_x = box_cenx*CELL_SIZE - (box_w/2)
    box_y = box_ceny*CELL_SIZE - (box_h/2)
    return [box_x+x_offset,box_y+y_offset , box_w, box_h]

# Non Maximun Suppression
def NMS(cls_scores,flat_pred_box, iou_threshold,score_threshold):
    cls_score_mat = copy.deepcopy(cls_scores)
    for i in range(len(cls_score_mat)):
        # sort by row (per one class)
        temp_arg = np.argsort( - cls_score_mat[i,:])
        sorted_scores = cls_score_mat[i, temp_arg] # sorting

        # nms algorithm
        for j in range(len(sorted_scores)):
            if sorted_scores[j] == 0:
                continue

            # j is 0~97 , 49*2 boxes
            # change j to cell, box id
            cell_id = temp_arg[j] % (SEG_SIZE**2) # 0~48
            box_id = temp_arg[j] // (SEG_SIZE**2) # 0,1
            bbox_max = add_offset(flat_pred_box[cell_id, box_id*5 : box_id*5+4], cell_id) # get box's x, y, w, h
            
            for k in range(len(cls_score_mat[i,j+1:])-1):
                cell_id2 = temp_arg[k+j+1] % (SEG_SIZE**2)
                box_id2 = temp_arg[k+j+1] // (SEG_SIZE**2)
                bbox_cur = add_offset(flat_pred_box[cell_id2, box_id2*5 : box_id2*5+4], cell_id2)
                if calc_iou(bbox_max, bbox_cur) > iou_threshold :
                    sorted_scores[j+k+1] = 0
                    cls_score_mat[i, temp_arg[k+j+1]] = 0

    #
    cls_score_mat[cls_score_mat < score_threshold] = 0
    return cls_score_mat

# inference 
def detection_procedure(pred_boxes,th1= 0.3, th2 = 0.5, th3= 0.0):
    # 7,7,22 --> 49,22
    flat_pred_box = np.reshape(pred_boxes, [-1, pred_boxes.shape[-1]])

    cls_score_mat = np.concatenate([flat_pred_box[:,4:5] * flat_pred_box[:,10:], flat_pred_box[:,9:10] * flat_pred_box[:,10:]] ,0)
    cls_score_mat = np.transpose(cls_score_mat) 
    
    # thresholding 
    temp_ind = cls_score_mat < th1
    cls_score_mat[temp_ind] = 0

    # Non Maximun Suppression
    ## return cls_score_mat
    nms_result =  NMS(cls_score_mat,flat_pred_box, th2,th3)
    
    detect_obj = list()
    ind =0 
    for cls_per_box in nms_result.T:
        if np.max(cls_per_box) > 0:
            box_id = ind // SEG_SIZE**2
            cell_id = ind % SEG_SIZE**2
            bbox = flat_pred_box[cell_id, box_id*5 : box_id*5+4] # get box's x,y,w,h
            bbox = add_offset(bbox, cell_id)
            category_id = np.argmax(cls_per_box)
            detect_obj.append(bbox+[category_id])         

        ind += 1
    return detect_obj



if __name__ =="__main__":
    parser = build_parser()
    args = parser.parse_args()
    # print args
    for arg in vars(args):
        print(arg,':', getattr(args, arg))

    # make dir for result saving
    if not os.path.exists(args.test_result_path):
        os.makedirs(args.test_result_path)
    
    
    with tf.device('/gpu:'+args.gpu_num): 
        inp_img = tf.placeholder(name='input_image',shape=[None,IMAGE_SIZE, IMAGE_SIZE, 3], dtype= tf.float32)
        tout = yolo_net(inp_img)

    
    # get vgg for style's weights
    reader = tf.train.NewCheckpointReader(args.model_load_path)
    restore_dict = dict()
    for v in tf.global_variables():
        tensor_name= v.name.split(':')[0]
        if reader.has_tensor(tensor_name):
            print('has tensor : ',tensor_name)
            restore_dict[tensor_name] = v
    #
    saver = tf.train.Saver(restore_dict)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, args.model_load_path)

    
    feed_forward_times = []
    detect_times = []

    test_image_names = os.listdir(args.test_image_path)
    for name in test_image_names:
        img_name = args.test_image_path+name
        img = read_img(img_name)
        img = np.expand_dims(img, axis=0)
        
        t1 = time.time()
        test_tout = sess.run(tout, {inp_img:img})
        feed_forward_times.append(time.time()-t1)

        t1 = time.time()
        detected = detection_procedure(test_tout, th1= args.threshold1, th2 = args.threshold2, th3=args.threshold3)
        detect_times.append(time.time()- t1)
        detected_name = args.test_result_path+'detected_'+name
        plot_img_bbox(img[0,...], detected, detected_name)

    print("%s: Average feed forward time : %2.4f, Average detection procedure time : %2.4f"%(datetime.now(),np.mean(feed_forward_times), np.mean(detect_times)))

    
        
    