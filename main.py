import cv2
import mediapipe as mo

mp_drawing=mo.solutions.drawing_utils
mp_face_mesh=mo.solutions.face_mesh
drawing_spec=mo.DrawingSpec(thickness=1,
circle_radius=3)

video=cv2.VideoCapture(0)

with mo_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)as face_mesh:
    while  True :
        ret,image=video.read()
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image.flags.writeable=True
        results=face_mesh.process(image)
        #print(results)
        image.flags.writeable=False
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mo_drawing.draw_landmarks(image=image,
                                        landmark_list= face_landmarks,
                                         connections=mo_face_mesh.FACE_CONNECTIONS,
                                         landmark_drawing_spec=drawing_spec,
                                         connection_drawing_spec=drawing_spec )
        cv2.imshow("Face Mesh",image)
        k=cv2.waitKey(1)
        if k==ord('c') :
            break
    video.realease()
    cv2.destroyAllWindows()        