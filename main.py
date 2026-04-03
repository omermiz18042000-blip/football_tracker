import cv2

# פותחים את קובץ הוידאו
video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video. Check the file name.")

# מתחילים לולאה שקוראת את הוידאו פריים אחר פריים
while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("End of video.")
        break
        
    # מציגים את הפריים בחלון שנקרא 'Football'
    cv2.imshow('Football', frame)
    
    # הוידאו ירוץ, ואם נלחץ על האות q במקלדת - הוא ייעצר
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# בסוף, מנקים הכל וסוגרים חלונות
cap.release()
cv2.destroyAllWindows()