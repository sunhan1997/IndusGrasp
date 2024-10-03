import cv2
import torch.utils.data




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load('/home/sunh/6D_grasp/IndusGrasp/maskrcnn/model_50.pth')
# move model to the right device
model.to(device)
model.eval()


for i in range(0,20):
    # read image and process image
    # img = cv2.imread('/media/sunh/Samsung_T5/6D_data/my_6d/6D_PanelPose/work_space/data/panel7/JPEGImages/{}.jpg'.format(idx))
    img = cv2.imread('/home/sunh/6D_grasp/IndusGrasp/maskrcnn/502/ori/test_{}.png'.format(i))
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = img_show.copy()
    img = torch.from_numpy(img_show.transpose((2, 0, 1)))
    img = img.float().div(255)

    # prediction
    with torch.no_grad():
        prediction = model([img.to(device)])
        # print(prediction)

    # procrss prediction
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks = prediction[0]['masks']
    names = {'0': 'background', '1': 'panel'}

    masks_sunhan = masks[0].cpu().numpy()
    masks_sunhan = masks_sunhan[0]
    masks_sunhan[masks_sunhan > 0] = 255
    result[:,:,0][masks_sunhan>0] = 255

    for idx in range(boxes.shape[0]):
        first_score = scores[0]
        if scores[idx] >= first_score:
            first_score = scores[idx]
            x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])

    result_crop = result[int(y1):int(y2), int(x1): int(x2)]
    cv2.imwrite('/home/sunh/6D_grasp/IndusGrasp/maskrcnn/502/result/{}.png'.format(i), result)

    # cv2.imshow('masks_sunhan', masks_sunhan)
    # cv2.imshow('result', result)
    # cv2.imshow('result_crop', result_crop)
    # cv2.waitKey()

    # m_bOK = False
    # for idx in range(boxes.shape[0]):
    #     if scores[idx] >= 0.8:
    #         m_bOK = True
    #         color = random_color()
    #         mask = masks[idx, 0].mul(255).byte().cpu().numpy()
    #         thresh = mask
    #         contours, hierarchy = cv2_util.findContours(
    #             thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    #         )
    #         cv2.drawContours(img_show, contours, -1, color, -1)
    #
    #         x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
    #         name = names.get(str(labels[idx].item()))
    #         cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
    #         cv2.putText(result, text=name, org=(int(x1), int(y1) + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                     fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)
    #         dst1 = cv2.addWeighted(result, 0.7, img_show, 0.5, 0)
    #
    # if m_bOK:
    #     cv2.imshow('result', dst1)
    #     cv2.waitKey()

