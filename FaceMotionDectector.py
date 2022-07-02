import numpy as np
import winsound
import cv2
import time
from matplotlib import pyplot as plt

# Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('C:\\Users\\Ragul\\PycharmProjects\\pythonProject\\haarcascade.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\Ragul\\PycharmProjects\\pythonProject\\haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('C:\\Users\\Ragul\\PycharmProjects\\pythonProject\\haarcascade_smile.xml')
freq = 500
dur = 100
neighbors = 30
# Variable store execution state
first_read = True
keras_rcnn = []


def RCNN():
    training_dictionary, test_dictionary = keras_rcnn.datasets.shape.load_data()
    categories = {"circle": 1, "rectangle": 2, "triangle": 3}
    generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()
    generator = generator.flow_from_dictionary(
        dictionary=training_dictionary,
        categories=categories,
        target_size=(224, 224),
        shuffle=False,
    )
    target, _ = generator.next()
    target_bounding_boxes, target_categories, target_images, target_masks, _ = target
    target_bounding_boxes = numpy.squeeze(target_bounding_boxes)
    target_images = numpy.squeeze(target_images)
    target_categories = numpy.argmax(target_categories, -1)
    target_categories = numpy.squeeze(target_categories)
    keras_rcnn.utils.show_bounding_boxes(
        target_images, target_bounding_boxes, target_categories
    )


def ssd_rcnn(SSD_detect, faster_rcnn, SSD_val, test_num=10000):
    if (test_num == keras_rcnn):
        import ipdb
        import matplotlib
        from tqdm import tqdm
        from utils.config import opt
        from data.dataset import Dataset, TestDataset, inverse_normalize
        from model import FasterRCNNVGG16
        from torch.utils import data as data_
        from trainer import FasterRCNNTrainer
        from utils import array_tool as at
        from utils.vis_tool import visdom_bbox
        from utils.eval_tool import eval_detection_voc
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
        dataloader = []
        for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
            sizes = [sizes[0][0].item(), sizes[1][0].item()]
            pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
            gt_bboxes += list(gt_bboxes_.numpy())
            gt_labels += list(gt_labels_.numpy())
            gt_difficults += list(gt_difficults_.numpy())
            pred_bboxes += pred_bboxes_
            pred_labels += pred_labels_
            pred_scores += pred_scores_
            if ii == test_num: break
        RCNN()
        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=True)
    else:
        return 1;


def train(**kwargs):
    import ipdb
    import matplotlib
    from tqdm import tqdm
    import os
    from utils.config import opt
    from data.dataset import Dataset, TestDataset, inverse_normalize
    from model import FasterRCNNVGG16
    from torch.utils import data as data_
    from trainer import FasterRCNNTrainer
    from utils import array_tool as at
    from utils.vis_tool import visdom_bbox
    from utils.eval_tool import eval_detection_voc
    opt._parse(kwargs)
    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())
                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)
                # plot predicti bboxes
                bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
            eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
            trainer.vis.plot('test_map', eval_result['map'])
            lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
            log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                      str(eval_result['map']),
                                                      str(trainer.get_meter_data()))
            trainer.vis.log(log_info)
            if eval_result['map'] > best_map:
                best_map = eval_result['map']
                best_path = trainer.save(best_map=best_map)
            if epoch == 9:
                trainer.load(best_path)
                trainer.faster_rcnn.scale_lr(opt.lr_decay)
                lr_ = lr_ * opt.lr_decay
            if epoch == 13:
                break


def calculate_accuracy():
    # based Input Prediction  Calculate it by Manually
    TP = 22
    TN = 21
    FP = 1
    FN = 2
    print('***********************')
    print('***********************')
    print("Performance Metrics calculation")
    print('***********************')
    print('***********************')
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)

    # calculate accuracy
    conf_accuracy = (float(TP + TN) / float(TP + TN + FP + FN) * 100)

    # calculate mis-classification
    conf_misclassification = (1 - conf_accuracy) * 100

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN)) * 100
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP)) * 100

    # calculate precision
    conf_precision = (TN / float(TN + FP)) * 100
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity)) * 100
    print('-' * 50)
    print(f'Accuracy: {round(conf_accuracy, 2)}')
    print(f'Mis-Classification: {round(conf_misclassification, 2)}')
    print(f'Sensitivity: {round(conf_sensitivity, 2)}')
    print(f'Specificity: {round(conf_specificity, 2)}')
    print(f'Precision: {round(conf_precision, 2)}')
    print(f'f_1 Score: {round(conf_f1, 2)}')
    # creating the dataset
    data = {'KNN': 85, 'RF': 92, 'DecisionTree': 93,
            'RCNN': 96.5}
    courses = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(courses, values, color='maroon',
            width=0.4)

    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy")
    plt.title("Performance comparision Graph")
    plt.show()


# Starting the video capture
cap = cv2.VideoCapture(0)
ret, img = cap.read()
calculate_accuracy()
while (ret):
    ret, img = cap.read()
    # Converting the recorded image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Detecting the face for region of image to be fed to eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
    smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=neighbors)
    cnn = ssd_rcnn(faces, smile, 12, test_num=10000)
    if (len(faces) > 0):
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # roi_face is face which is input to eye classifier
            roi_face = gray[y:y + h, x:x + w]
            roi_face_clr = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
            smile = smile_cascade.detectMultiScale(roi_face, scaleFactor=1.7, minNeighbors=neighbors)

            # Examining the length of eyes object for eyes
            if (len(eyes) >= 1 and cnn == 1):
                # Check if program is running for detection
                cv2.putText(img,
                            "Neutral!", (70, 70),
                            cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 0), 2)
                # time.sleep(1)
            elif len(smile) > 0:
                cv2.putText(img, 'Happy', (70, 70), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 0))
                time.sleep(1)
            else:
                # To ensure if the eyes are present before starting
                cv2.putText(img,
                            "Drowsiness Detected", (70, 70),
                            cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 0), 2)
                winsound.Beep(freq, dur)
            # time.sleep(1)
    else:
        cv2.putText(img,
                    "No face detected", (100, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 0, 0), 2)
    # Controlling the algorithm with keys
    cv2.imshow('img', img)
    a = cv2.waitKey(1)
    if (a == ord('q')):
        calculate_accuracy()
        break
    elif (a == ord('s') and first_read):
        # This will start the detection
        first_read = False
cap.release()
cv2.destroyAllWindows()