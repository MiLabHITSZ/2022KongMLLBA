import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append('../yolo_master')
import logging
import torch
from yolo_master.yolo_voc2012 import *

class MLLBA(object):
    def __init__(self, model):
        self.model = model

    def generate_np(self, x_list, ori_loader, **kwargs):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        logging.info('prepare attack')
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']
        alpha = kwargs['alpha']
        pop_size = kwargs['pop_size']
        batch_size = kwargs['batch_size']
        yolonet = kwargs['yolonet']
        x_adv = []
        success = 0
        nchannels,img_rows, img_cols,  = x_list.shape[1:4]
        count = 0
        l2_sum=0
        fail = []
        np.random.seed(1)
        for i in range(len(x_list)):
                best_l2=999
                try_time=0
                giveup =0
                threshold = 100
                queries = 0
                while try_time < 50:
                    istoobig = 1
                    while (istoobig == 1):
                        #Generate initial adversarial examples
                        Adv, isAder,giveup = GenBig(self.model, x_list[i], y_target[i],ori_loader,try_time,yolonet,giveup)
                        if isAder == False :
                            break

                        #Optimizing perturbations by differential evolution
                        r, queries_tem, l2 ,istoobig= DE(pop_size, img_rows * img_cols * nchannels, self.model, x_list[i],
                                              alpha, batch_size, y_target[i], Adv,giveup,threshold)
                        queries += queries_tem
                    if isAder == False:
                        x_adv_tem = np.clip(x_list[i], 0, 1)
                        best_l2 = 0
                        break
                    x_adv_loop = np.clip(x_list[i] + np.reshape(r, x_list.shape[1:]) , 0, 1)
                    l2 = np.linalg.norm(x_adv_loop-x_list[i])
                    if l2 <best_l2:
                        best_l2 = l2
                        x_adv_tem = x_adv_loop
                    if l2 <= 50:
                        break
                    else:
                        logging.info("The number of failed attempts:"+str(try_time)+
                                     ". The l2 norm of the adversarial samples generated this time is: "+str(l2)+
                                     ". The l2 norm of the current best adversarial example is: "+str(best_l2))
                        try_time+=1
                count += 1
                if best_l2 >=100:
                    x_adv_tem = np.clip(x_list[i], 0, 1)
                    best_l2 = 0
                l2_sum += best_l2
                with torch.no_grad():
                    if torch.cuda.is_available():
                        adv_pred = self.model(
                            torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32).cuda()).cpu()
                    else:
                        adv_pred = self.model(torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32))
                adv_pred = np.asarray(adv_pred)
                pred = adv_pred.copy()
                pred[pred >= (0.5 + 0)] = 1
                pred[pred < (0.5 + 0)] = -1
                adv_pred_match_target = np.all((pred == y_target[i]), axis=1)
                if adv_pred_match_target:
                    success = success + 1
                else:
                    fail.append(i)
                    logging.info('Attack fail：')
                    logging.info(fail)
                x_adv.append(x_adv_tem)

                if success!=0:
                    logging.info('Current progress: ' + str(count) + '/' + str(batch_size))
                    logging.info('Number of successful attacks: ' + str(success))
                    logging.info('Current success rate: ' + str(success / count))
                    logging.info("The l2 norm of the current adversarial example: "+str(best_l2))
                    logging.info("The average l2 norm: "+str(l2_sum/success))
                else:
                    logging.info('Current progress: ' + str(count) + '/' + str(batch_size))
                    logging.info('Number of successful attacks: ' + str(success) )
                    logging.info('Current success rate: ' + str(success / count) )
                    logging.info("The l2 norm of the current adversarial example: " + str(best_l2))
        return x_adv , queries

class Problem:
    def __init__(self, model, adv_img, target, alpha, batch_size, ori_img):
        self.model = model
        self.adv_img = adv_img
        self.target = target
        self.alpha = alpha
        self.batch_size = batch_size
        self.ori_img = ori_img

    def setalpha(self,alpha):
        self.alpha = alpha

    def evaluate(self, x, pur_abs):
        x= x * self.alpha
        x_abs = np.abs(x)
        for i in range(len(x_abs)):
            x_abs[i] = np.clip(x_abs[i],0., pur_abs[i])
        x= x_abs*np.sign(x)
        with torch.no_grad():
            adv = np.clip(np.tile(self.adv_img, (len(x), 1, 1, 1)) + np.reshape(x, (len(x),) + self.adv_img.shape), 0., 1.)
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(adv, dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(adv, dtype=torch.float32))

        ori_pred = np.asarray(predict)
        pred = ori_pred.copy()
        pred[pred >= (0.5 + 0)] = 1
        pred[pred < (0.5 + 0)] = -1
        pop_target = np.tile(self.target, (len(x), 1))
        pred_no_match = []
        for i in range(len(pred)):
            if np.any(pred[i] != pop_target[i]):
                pred_no_match.append(i)
        pur = (adv-np.clip(np.tile(self.ori_img, (len(x), 1, 1, 1)),0.,1.))
        pur = pur.reshape(pur.shape[0],-1)
        fitness = np.linalg.norm(pur,axis=1, ord=2)
        if(len(pred_no_match)!=0):
            fitness[pred_no_match] = 999
        return fitness

def mating(pop,F):

    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    mutation = pop + F * (p2 - p3)
    return mutation

def mating_best(pop,fitness,F):
    best = np.argmin(fitness)
    mutation= np.copy(pop)
    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    for i in range(len(pop)):
        mutation[i] = pop[best] + F * (p2[i]-p3[i])
    return  mutation


def select(pop,fitness,off,off_fitness):
   new_pop = pop.copy()
   new_fitness = fitness.copy()
   i=np.argwhere(fitness>off_fitness)
   new_pop[i] = off[i].copy()
   new_fitness[i] = off_fitness[i].copy()
   return new_pop ,new_fitness

def meanpop(pop,sort,num):
    sum = pop[0]*0
    for j in range(num):
        i = sort[j]
        sum = sum+pop[i]
    mean = sum / num
    return mean

def DE(pop_size, length, model, ori_image, alpha, batch_size, target,Adv,giveup,threshold):
    generation_save = np.zeros((10000,))
    curr_alpha = alpha
    problem = Problem(model, Adv, target, alpha, batch_size, ori_image)
    pop = np.random.uniform(0, 1, size=(pop_size, length))
    pur = Adv-ori_image
    pur=pur.reshape(-1)
    sin_pur = pur
    sin_pur_abs = np.abs(pur)
    pur=pur[np.newaxis,:]
    pur=np.tile(pur, (pop_size, 1))
    pur_abs = np.abs(pur)
    for i in range(len(pop)):
        pop[i]=np.random.uniform(0, pur_abs[i])
    pop = -1*pop*np.sign(pur)
    eval_count = 0
    fitness= problem.evaluate(pop,pur_abs)
    eval_count += pop_size
    count = 0
    fitmin = np.min(fitness)
    minl2=np.linalg.norm(Adv-ori_image)
    logging.info("The l2 norm of the initial adversarial example: "+str(minl2))
    generation_save[count] = fitmin
    last_fit = fitmin
    F = 0.5
    count_end=0
    count_rand=0
    randflag = 0
    best = np.argmin(fitness)
    x = pop[best] * curr_alpha*0
    if minl2 >= threshold and giveup==0:
        logging.info("The initial l2 norm is too large, try to regenerate an initial adversarial example.")
        return sin_pur + x, eval_count, minl2, 1
    while (eval_count < 200 *pop_size):
            count += 1
            if randflag == 1 :
                off = mating(pop, F)
            else:
                off = mating_best(pop,fitness,F)
            off_fitness  = problem.evaluate(off,pur_abs)
            fitmin = np.min(off_fitness)
            if fitmin == 999:
                if(curr_alpha<=1):
                    logging.info("After "+str(count) +" times, the optimization stops. The current minimum l2 norm is " + str(minl2))
                    break
                else:
                    curr_alpha = curr_alpha*0.8
                    problem.setalpha(curr_alpha)
                    logging.info("Appropriately reduce the alpha to " +str(curr_alpha))

                    pop = np.random.uniform(0, 1, size=(pop_size, length))
                    pur = Adv - ori_image
                    pur = pur.reshape(-1)
                    pur = pur[np.newaxis, :]
                    pur = np.tile(pur, (pop_size, 1))
                    pur_abs = np.abs(pur)
                    for i in range(len(pop)):
                        pop[i] = np.random.uniform(0, pur_abs[i])
                    pop = -1 * pop * np.sign(pur)

                    fitness = problem.evaluate(pop, pur_abs)
                    continue
            eval_count += pop_size
            pop ,fitness = select (pop,fitness,off,off_fitness)
            fitmin = np.min(fitness)
            if(fitmin<minl2 ):
                best = np.argmin(fitness)
                x = pop[best] * curr_alpha
                minl2 = fitmin
            generation_save[count] = fitmin
            if (count == 10 and fitmin >= 100):
                break
            if (fitmin <= 1):
                logging.info("The perturbation is invisible enough, the evolution terminates.")
                break
            if (fitmin == last_fit):
                count_end += 1
                if (count_end == 11):
                    logging.info("The optimal fitness remained unchanged for 10 consecutive generations, the evolution terminated early.")
                    break
            else:
                last_fit = fitmin
                count_end = 0
            if (curr_alpha < 0.5):
                logging.info("Alpha is too small, the evolution terminates.")
                break
            if(last_fit-fitmin)<0.5 :
                count_rand+=1
                if (count_rand == 10):
                    if fitmin > minl2:
                        break
                    curr_alpha = curr_alpha * 0.8
                    problem.setalpha(curr_alpha)
                    logging.info("The change in the optimal fitness for 10 consecutive generations is too small, so adjust alpha to " + str(curr_alpha))
                    count_rand = 0
                    pop = np.random.uniform(0, 1, size=(pop_size, length))
                    pur = Adv - ori_image
                    pur = pur.reshape(-1)
                    pur = pur[np.newaxis, :]
                    pur = np.tile(pur, (pop_size, 1))
                    pur_abs = np.abs(pur)
                    for i in range(len(pop)):
                        pop[i] = np.random.uniform(0, pur_abs[i])
                    pop = -1 * pop * np.sign(pur)

                    fitness = problem.evaluate(pop, pur_abs)
            else:
                count_rand=0

    x_abs = np.abs(x)
    for i in range(len(x_abs)):
        x_abs[i] = np.clip(x_abs[i], 0., sin_pur_abs[i])
    x = x_abs * np.sign(x)
    return sin_pur+x, eval_count, minl2, 0

def IOU(put,box):
    s_box = (box[2] - box[0]) * (box[3] - box[1])
    s_put = ( put[2]-put[0])*(put[3]-put[1])
    s_overlap = (min(box[3],put[3])-max(box[1],put[1]))*(min(box[2],put[2])-max(box[0],put[0]))
    iou = max(0, s_overlap / (s_box + s_put - s_overlap))
    return iou

def GenBig( model, image, target,ori_loader,jump,yolonet,quit):
    img_dir = "../data/voc2012/VOCdevkit/VOC2012/JPEGImages/"
    imglist = os.listdir(img_dir)
    boxes = []
    target_label = np.argwhere(target > 0)
    time = 0
    giveup = quit
    if quit !=1:
        while time in range(200):
            x_adv = np.transpose(image, (1, 2, 0))
            x_adv = Image.fromarray(np.uint8(x_adv * 255))
            for num in range(len(target_label)):
                flag = target_label[num][0]
                file = open("../data/voclabels.txt")
                for i in range(flag):
                    skip = file.readline()
                lab = file.readline().split('\n')[0]
                seed = np.random.uniform(0,9000)
                count = 0
                for img_name in imglist:
                    if count<seed:
                        count+=1
                        continue
                    img = Image.open('../data/voc2012/VOCdevkit/VOC2012/JPEGImages/' + img_name)
                    windows=yolonet.detect_image(img)
                    find = 0
                    for win in  windows:
                        if win[0] == lab:
                            pos = []
                            pos.append(win[1])
                            pos.append(win[2])
                            pos.append(win[3])
                            pos.append(win[4])
                            find=1
                            break
                    if find==1:
                        part = img.crop(pos)
                        break
                for i in range(20):
                    x1 = int(np.random.uniform(448-part.size[0]))
                    y1 = int(np.random.uniform(448-part.size[1]))
                    put = [x1, y1,x1+part.size[0],y1+part.size[1]]
                    can = True
                    for box in boxes:
                        if (put[0]>box[0] and put[0]<box[2] ) or (put[1]>box[1] and put[1]<box[3]):
                            can = False
                    if can == True:
                        boxes.append(put)
                        break
                x_adv.paste(part, put)
            x_adv = np.asarray(x_adv)/255.
            x_adv = np.transpose(x_adv, (2, 0, 1))
            x_adv = np.clip(np.tile(x_adv, (1, 1, 1, 1)), 0., 1.)
            with torch.no_grad():
                if torch.cuda.is_available():
                    predict = model(torch.tensor(x_adv, dtype=torch.float32).cuda()).cpu()
                else:
                    predict = model(torch.tensor(x_adv, dtype=torch.float32))
            ori_pred = np.asarray(predict)
            pred = ori_pred[0].copy()
            pred[pred >= (0.5 + 0)] = 1
            pred[pred < (0.5 + 0)] = -1
            adv_pred_match_target = np.all((pred == target))
            if adv_pred_match_target:
                x_adv = x_adv.squeeze()
                logging.info(("Successfully generated initial adversarial example"))
                return x_adv ,True, giveup
            else:
                time+=1

    # Unable to generate initial adversarial samples by YOLO object detector
    # Try directly select initial adversarial examples from the surrogate dataset
    giveup=1
    logging.info("Unable to generate initial adversarial samples by YOLO object detector")
    count = 0
    for i, (input, label) in enumerate(ori_loader):
        x_list = input[0].cpu().numpy()
        label = np.asarray(label)
        for j in range(label.shape[0]):
            adv_pred_match_target = np.all((label[j] == target))
            if adv_pred_match_target:
                if count < jump:
                    count += 1
                    continue
                logging.info('Successfully found an initial adversarial example in surrogate dataset')
                return x_list[j], True,giveup
    return x_list[0], False,giveup


