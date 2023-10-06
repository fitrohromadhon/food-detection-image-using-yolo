from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.pt")
img = cv2.imread("uji/pisang51.jpg")
img = cv2.resize(img, (640,640))

classes = []
with open("classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

kaloriapel = 72
kaloriapel2 = 36
kaloriayam = 239
kalorijeruk = 62
kalorijeruk2 = 31
kalorinasi = 204
kaloriperkedel = 21
kaloripisang = 105
kaloritahu = 35
kaloridadar = 93
kaloritelur = 77
kaloritelur2 = 39
kaloritempe = 34

totalApelCount = []
totalApel2Count = []
totalAyamCount = []
totalJerukCount = []
totalJeruk2Count = []
totalNasiCount = []
totalPerkedelCount = []
totalPisangCount = []
totalTahuCount = []
totalDadarCount = []
totalTelurCount = []
totalTelur2Count = []
totalTempeCount = []

results = model(img)
result = results[0]

bboxes = np.array(result.boxes.xyxy.cpu(), dtype=int)
class_id = np.array(result.boxes.cls.cpu(), dtype=int)

for cls, bbox in zip (class_id, bboxes):
    x, y, x2, y2 = bbox
    class_name = classes[cls]
    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 225), 2)
    cv2.putText(img, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    if class_name == "Apel":
        cv2.putText(img, (str(kaloriapel)+(" kal")), (x + 45, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalApelCount.append(class_name)
    if class_name == "Apel Set":
        cv2.putText(img, (str(kaloriapel2)+(" kal")), (x + 85, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalApelCount.append(class_name)
    if class_name == "Ayam Goreng":
        cv2.putText(img, (str(kaloriayam)+(" kal")), (x + 120, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalAyamCount.append(class_name)
    if class_name == "Jeruk":
        cv2.putText(img, (str(kalorijeruk)+(" kal")), (x + 50, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalJerukCount.append(class_name)
    if class_name == "Jeruk Set":
        cv2.putText(img, (str(kalorijeruk2)+(" kal")), (x + 90, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalJerukCount.append(class_name)
    if class_name == "Nasi Putih":
        cv2.putText(img, (str(kalorinasi)+(" kal")), (x + 90, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalNasiCount.append(class_name)
    if class_name == "Perkedel":
        cv2.putText(img, (str(kaloriperkedel)+(" kal")), (x + 80, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalPerkedelCount.append(class_name)
    if class_name == "Pisang":
        cv2.putText(img, (str(kaloripisang)+(" kal")), (x + 60, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalPisangCount.append(class_name)
    if class_name == "Tahu Goreng":
        cv2.putText(img, (str(kaloritahu)+(" kal")), (x + 115, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalTahuCount.append(class_name)
    if class_name == "Telur Dadar":
        cv2.putText(img, (str(kaloridadar)+(" kal")), (x + 105, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalDadarCount.append(class_name)
    if class_name == "Telur Rebus":
        cv2.putText(img, (str(kaloritelur) + (" kal")), (x + 105, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalTelurCount.append(class_name)
    if class_name == "Telur Rebus Set":
        cv2.putText(img, (str(kaloritelur2)+(" kal")), (x + 145, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalTelurCount.append(class_name)
    if class_name == "Tempe Goreng":
        cv2.putText(img, (str(kaloritempe)+(" kal")), (x + 130, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        totalTempeCount.append(class_name)

#cv2.putText(img, "Kalori Apel: " + str(len(totalApelCount)*72), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
#cv2.putText(img, "Kalori Ayam: " + str(len(totalAyamCount)*239), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
#cv2.putText(img, "Kalori Jeruk: " + str(len(totalJerukCount)*62), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
#cv2.putText(img, "Kalori Nasi: " + str(len(totalNasiCount)*204), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
#cv2.putText(img, "Kalori Perkedel: " + str(len(totalPerkedelCount)*21), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
#cv2.putText(img, "Kalori Pisang: " + str(len(totalPisangCount)*105), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
#cv2.putText(img, "Kalori Tahu: " + str(len(totalTahuCount)*35), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
#cv2.putText(img, "Kalori T.Dadar: " + str(len(totalDadarCount)*93), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
#cv2.putText(img, "Kalori T.Rebus: " + str(len(totalTelurCount)*77), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
#cv2.putText(img, "Kalori Tempe: " + str(len(totalTempeCount)*34), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

TotalKalori = str((len(totalApelCount)*72)+(len(totalApel2Count)*36)+(len(totalAyamCount)*239)+(len(totalJerukCount)*62)+
                  (len(totalJeruk2Count)*31)+(len(totalNasiCount)*204)+(len(totalPerkedelCount)*21)+(len(totalPisangCount)*105)+
                  (len(totalTahuCount)*35)+(len(totalDadarCount)*93)+(len(totalTelurCount)*77)+(len(totalTelur2Count)*39)+(len(totalTempeCount)*34))

cv2.putText(img, "TOTAL KALORI: " + str(TotalKalori) + " kal", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()