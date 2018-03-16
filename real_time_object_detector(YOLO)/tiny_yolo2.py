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
    parser.add_argument('--epochs', type=int,
                        help='num epochs', default = 100)
        
    parser.add_argument('--batch-size',type=int,
                        help='training batch size',default=8)

    parser.add_argument('--model-save-path', type=str,
                        help='model save path',default='/home/minsung/yolo_model/')
    
    parser.add_argument('--lr', type=float,
                        help='training learning rate', default=0.00001)
    
    parser.add_argument('--gpu-num', type=str,
                        help='number of gpu to use')

    parser.add_argument('--test-result-path', type=str,
                        help='image result during training',default='yolo_test_result2/')
    
    parser.add_argument('--noobject-weight', type=float,
                        help='no object loss weights', default=0.5)
    
    parser.add_argument('--coord-weight', type=float,
                        help='coordinate loss weight', default=5)

    
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

def read_img(img_path,IMAGE_SIZE):
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
def detection_procedure(pred_boxes,th1= 0.2, th2 = 0.5, th3= 0.0):
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
        truth_boxes = tf.placeholder(name='normalize_bboxes',shape= [None,SEG_SIZE,SEG_SIZE, 22], dtype=tf.float32) # box's x,y,w,h,confidence, one hot class

        tout = yolo_net(inp_img)

        obj_I = tf.placeholder(shape=[None,SEG_SIZE,SEG_SIZE, 1], dtype=tf.float32,name='object_in_cell')
        no_I = tf.placeholder(shape=[None,SEG_SIZE,SEG_SIZE, 1], dtype=tf.float32,name='noobject_in_cell')


        obj_yolo_out = tout*obj_I
        noobj_yolo_out = tout*no_I

        noobject_loss = tf.nn.l2_loss(noobj_yolo_out[:,:,:,4]) + tf.nn.l2_loss(noobj_yolo_out[:,:,:,9])

        # intersection box
        ax = tf.reduce_max(tf.stack([obj_yolo_out[:,:,:,0], truth_boxes[:,:,:,0]],axis=3), axis=3)
        ay = tf.reduce_max(tf.stack([obj_yolo_out[:,:,:,1], truth_boxes[:,:,:,1]], axis=3), axis=3)
        aw = tf.reduce_min(tf.stack([obj_yolo_out[:,:,:,2], truth_boxes[:,:,:,2]], axis=3), axis=3)- ax
        ah = tf.reduce_min(tf.stack([obj_yolo_out[:,:,:,3], truth_boxes[:,:,:,3]],axis=3), axis=3) - ay
        a_intersection_size = aw*ah

        bx = tf.reduce_max(tf.stack([obj_yolo_out[:,:,:,5], truth_boxes[:,:,:,0]],axis=3),axis=3)
        by = tf.reduce_max(tf.stack([obj_yolo_out[:,:,:,6], truth_boxes[:,:,:,1]], axis=3), axis=3)
        bw = tf.reduce_min(tf.stack([obj_yolo_out[:,:,:,7], truth_boxes[:,:,:,2]],axis=3),axis=3) - bx
        bh = tf.reduce_min(tf.stack([obj_yolo_out[:,:,:,8], truth_boxes[:,:,:,3]], axis=3),axis=3) - by
        b_intersection_size = bw*bh

        # box size
        a_box_size = tf.multiply(obj_yolo_out[:,:,:,2], obj_yolo_out[:,:,:,3])
        b_box_size = tf.multiply(obj_yolo_out[:,:,:,7], obj_yolo_out[:,:,:,8])

        truth_size = tf.multiply(truth_boxes[:,:,:,2], truth_boxes[:,:,:,3])


        # iou
        a_iou = a_intersection_size/(a_box_size+truth_size - a_intersection_size +1e-12)
        b_iou = b_intersection_size/(b_box_size+truth_size - b_intersection_size +1e-12)

        # responsible predction box
        responsible_b = tf.argmax(tf.stack([a_iou, b_iou], axis=3), axis=3)
        responsible_a = 1- responsible_b


        duplicated1 = tf.stack( [responsible_a,responsible_a,responsible_a,responsible_a,responsible_a], axis=3)
        duplicated1 = tf.cast(duplicated1, tf.float32)

        duplicated2 = tf.stack( [responsible_b,responsible_b,responsible_b,responsible_b,responsible_b],axis=3)
        duplicated2 = tf.cast(duplicated2, tf.float32)

        bs = tf.shape(tout)[0]
        cls1 = tf.ones(shape=[bs,SEG_SIZE,SEG_SIZE,12], dtype=tf.float32)
        cls0 = tf.zeros(shape=[bs,SEG_SIZE,SEG_SIZE,12],dtype=tf.float32)

        responsible = tf.concat([duplicated1,duplicated2, cls1],axis=3)*obj_I
        not_responsible= tf.concat([duplicated2, duplicated1,cls0],axis=3)*obj_I

        response_yolo_out= responsible*obj_yolo_out
        not_response_yolot_out = not_responsible*obj_yolo_out

        noobject_loss += tf.nn.l2_loss([not_response_yolot_out[:,:,:,4], not_response_yolot_out[:,:,:,9]])

        noobject_loss = noobject_loss*args.noobject_weight

        class_loss = tf.nn.l2_loss(truth_boxes[:,:,:,10:] - response_yolo_out[:,:,:,10:])
        temp_truth = truth_boxes*responsible

        coord_loss = tf.nn.l2_loss( [temp_truth[:,:,:,0:2] - response_yolo_out[:,:,:,0:2]]) # xy
        coord_loss += tf.nn.l2_loss( [temp_truth[:,:,:,5:7] - response_yolo_out[:,:,:,5:7]]) # xy

        coord_loss += tf.nn.l2_loss( [temp_truth[:,:,:,2:4] - response_yolo_out[:,:,:,2:4]]) # wh
        coord_loss += tf.nn.l2_loss( [temp_truth[:,:,:,7:9] - response_yolo_out[:,:,:,7:9]]) # wh

        coord_loss = args.coord_weight*coord_loss
        # 
        confiden_loss = tf.nn.l2_loss(temp_truth[:,:,:,4] - response_yolo_out[:,:,:,4])
        confiden_loss += tf.nn.l2_loss(temp_truth[:,:,:,9] - response_yolo_out[:,:,:,9])

        total_loss = coord_loss + noobject_loss + confiden_loss + class_loss

        train_step = tf.train.AdamOptimizer(args.lr).minimize(total_loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver  = tf.train.Saver()
    
    
    iteration_ind = 0
    batch_i = 0
    batch_img = []
    batch_ngbox = []
    batch_obj = []
    batch_noobj = []
    for epo in range(args.epochs):
        # for ind in range(len(coco_images_info))
        for _ in range(len(coco_images_info)//args.batch_size):
            coco_indx = np.random.randint(0, len(coco_images_info))
            img_info = coco_images_info[coco_indx]
            img, gboxes = get_coco_data(img_info)
            ngbox, object_in_cell, noobject_in_cell = assign_bbox(gboxes)
            # batch
            batch_img.append(img)
            batch_ngbox.append(ngbox)
            batch_obj.append(object_in_cell)
            batch_noobj.append(noobject_in_cell)
            batch_i += 1
            if batch_i % args.batch_size ==0 :
                bimg = np.array(batch_img)
                bimg = np.squeeze(bimg)
                bngbox = np.array(batch_ngbox)
                bobj = np.array(batch_obj)
                bnoobj = np.array(batch_noobj)

                feedict = {inp_img:bimg, truth_boxes:bngbox,obj_I:bobj, no_I:bnoobj}
                sess.run(train_step,feedict)
                iteration_ind += 1
                batch_img = []
                batch_ngbox = []
                batch_obj = []
                batch_noobj = []
                
                
                
        # test output image per epoch
        print(datetime.now(),': iteration : %d'%iteration_ind)
        _tout,_total_loss, _coord_loss, _noobject_loss, _confidence_loss, _class_loss = sess.run([tout,total_loss, coord_loss, noobject_loss, confiden_loss,class_loss],feedict)
        print(datetime.now(),': total loss : %e, coord Loss : %e, no Loss : %e, confi Loss : %e, class Loss : %e'%(_total_loss, _coord_loss, _noobject_loss, _confidence_loss, _class_loss))
        if not np.isnan(_total_loss):
            save_path = saver.save(sess, args.test_result_path+'yolo_tiny_model.ckpt')
            print(datetime.now(), ": Model saved in file : ", save_path)

        test_box = detection_procedure(_tout[0,...])
        test_name = args.test_result_path+str(epo)+'th_epoch_yolo_test_result.jpg'
        plot_img_bbox(img[0,:,:,:], test_box, test_name)



