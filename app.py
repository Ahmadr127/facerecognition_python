from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
from mysql.connector import Error

app = Flask(__name__, template_folder='templates', static_folder='static')

# Pastikan direktori yang diperlukan ada
os.makedirs('dataset', exist_ok=True)
os.makedirs('resources', exist_ok=True)

# Cek file cascade classifier
cascade_path = "resources/haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print(f"Error: File {cascade_path} tidak ditemukan!")
    print("Silakan download file dari: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
    exit(1)

def get_available_cameras():
    """Mendeteksi kamera yang tersedia"""
    available_cameras = []
    for i in range(10):  # Cek 10 kamera pertama
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="root",
            database="flask_db"
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

@app.route('/')
def home():
    try:
        connection = get_db_connection()
        if connection is None:
            return "Error: Tidak dapat terhubung ke database", 500
            
        mycursor = connection.cursor()
        mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
        data = mycursor.fetchall()
        mycursor.close()
        connection.close()
        return render_template('index.html', data=data)
    except Error as e:
        print(f"Error: {e}")
        return "Error: Terjadi kesalahan saat mengambil data", 500

@app.route('/addprsn')
def addprsn():
    try:
        connection = get_db_connection()
        if connection is None:
            return "Error: Tidak dapat terhubung ke database", 500
            
        mycursor = connection.cursor()
        mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
        row = mycursor.fetchone()
        nbr = row[0]
        mycursor.close()
        connection.close()
        return render_template('addprsn.html', newnbr=int(nbr))
    except Error as e:
        print(f"Error: {e}")
        return "Error: Terjadi kesalahan saat mengambil data", 500

@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    try:
        prsnbr = request.form.get('txtnbr')
        prsname = request.form.get('txtname')
        prsskill = request.form.get('optskill')

        connection = get_db_connection()
        if connection is None:
            return "Error: Tidak dapat terhubung ke database", 500
            
        mycursor = connection.cursor()
        mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                        (%s, %s, %s)""", (prsnbr, prsname, prsskill))
        connection.commit()
        mycursor.close()
        connection.close()
        return redirect(url_for('vfdataset_page', prs=prsnbr))
    except Error as e:
        print(f"Error: {e}")
        return "Error: Terjadi kesalahan saat menyimpan data", 500

@app.route('/edit/<id>')
def edit(id):
    try:
        connection = get_db_connection()
        if connection is None:
            return "Error: Tidak dapat terhubung ke database", 500
            
        mycursor = connection.cursor()
        mycursor.execute("SELECT * FROM prs_mstr WHERE prs_nbr = %s", (id,))
        data = mycursor.fetchone()
        mycursor.close()
        connection.close()
        return render_template('editprsn.html', data=data)
    except Error as e:
        print(f"Error: {e}")
        return "Error: Terjadi kesalahan saat mengambil data", 500

@app.route('/update/<id>', methods=['POST'])
def update(id):
    try:
        prsname = request.form.get('txtname')
        prsskill = request.form.get('optskill')
        prsactive = request.form.get('optactive')

        connection = get_db_connection()
        if connection is None:
            return "Error: Tidak dapat terhubung ke database", 500
            
        mycursor = connection.cursor()
        mycursor.execute("""UPDATE prs_mstr 
                           SET prs_name = %s, prs_skill = %s, prs_active = %s 
                           WHERE prs_nbr = %s""", 
                           (prsname, prsskill, prsactive, id))
        connection.commit()
        mycursor.close()
        connection.close()
        return redirect('/')
    except Error as e:
        print(f"Error: {e}")
        return "Error: Terjadi kesalahan saat mengupdate data", 500

@app.route('/delete/<id>')
def delete(id):
    try:
        connection = get_db_connection()
        if connection is None:
            return "Error: Tidak dapat terhubung ke database", 500
            
        mycursor = connection.cursor()
        mycursor.execute("DELETE FROM prs_mstr WHERE prs_nbr = %s", (id,))
        connection.commit()
        mycursor.close()
        connection.close()
        return redirect('/')
    except Error as e:
        print(f"Error: {e}")
        return "Error: Terjadi kesalahan saat menghapus data", 500

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    cameras = get_available_cameras()
    return render_template('gendataset.html', prs=prs, cameras=cameras)

@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    camera_id = request.args.get('camera', default=0, type=int)
    return Response(generate_dataset(nbr, camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_dataset(nbr, camera_id=0):
    face_classifier = cv2.CascadeClassifier(cascade_path)

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(int(camera_id))
    if not cap.isOpened():
        print(f"Error: Tidak dapat mengakses kamera {camera_id}")
        return "Error: Tidak dapat mengakses kamera!", 500

    try:
        connection = get_db_connection()
        if connection is None:
            return "Error: Tidak dapat terhubung ke database", 500
            
        mycursor = connection.cursor()
        mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
        row = mycursor.fetchone()
        lastid = row[0]

        img_id = lastid
        max_imgid = img_id + 100
        count_img = 0

        while True:
            ret, img = cap.read()
            if not ret:
                print(f"Error: Tidak dapat membaca frame dari kamera {camera_id}")
                break
                
            if face_cropped(img) is not None:
                count_img += 1
                img_id += 1
                face = cv2.resize(face_cropped(img), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = os.path.join("dataset", f"{nbr}.{img_id}.jpg")
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`, `img_path`) VALUES
                                    (%s, %s, %s)""", (img_id, nbr, file_name_path))
                connection.commit()

                frame = cv2.imencode('.jpg', face)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                    break
    except Error as e:
        print(f"Error: {e}")
        return "Error: Terjadi kesalahan saat mengambil data", 500
    finally:
        if 'mycursor' in locals():
            mycursor.close()
        if 'connection' in locals():
            connection.close()
        cap.release()
        cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "dataset"
    
    if not os.path.exists(dataset_dir):
        return "Error: Dataset directory tidak ditemukan!", 500

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        try:
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split(".")[1])

            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            continue

    if not faces:
        return "Error: Tidak ada gambar yang valid untuk training!", 500

    ids = np.array(ids)

    try:
        # Train the classifier and save
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")
        return redirect('../')
    except Exception as e:
        print(f"Error training classifier: {e}")
        return "Error: Gagal melatih classifier!", 500


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/countTodayScan')
def countTodayScan():
    try:
        connection = get_db_connection()
        if connection is None:
            return jsonify({'rowcount': 0})
            
        mycursor = connection.cursor()
        mycursor.execute("""SELECT COUNT(*) FROM accs_hist 
                           WHERE DATE(accs_date) = CURDATE()""")
        row = mycursor.fetchone()
        count = row[0]
        mycursor.close()
        connection.close()
        return jsonify({'rowcount': count})
    except Error as e:
        print(f"Error: {e}")
        return jsonify({'rowcount': 0})

@app.route('/loadData')
def loadData():
    try:
        connection = get_db_connection()
        if connection is None:
            return jsonify([])
            
        mycursor = connection.cursor()
        mycursor.execute("""SELECT * FROM accs_hist 
                           WHERE DATE(accs_date) = CURDATE() 
                           ORDER BY accs_date DESC""")
        data = mycursor.fetchall()
        mycursor.close()
        connection.close()
        return jsonify(data)
    except Error as e:
        print(f"Error: {e}")
        return jsonify([])

@app.route('/test_recognition')
def test_recognition():
    cameras = get_available_cameras()
    return render_template('fr_page.html', cameras=cameras)

@app.route('/video_feed')
def video_feed():
    camera_id = request.args.get('camera', default=0, type=int)
    return Response(face_recognition(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def face_recognition(camera_id=0):
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            try:
                connection = get_db_connection()
                if connection is None:
                    s = 'UNKNOWN'
                else:
                    mycursor = connection.cursor()
                    mycursor.execute("select b.prs_name "
                                 "  from img_dataset a "
                                 "  left join prs_mstr b on a.img_person = b.prs_nbr "
                                 " where img_id = " + str(id))
                    result = mycursor.fetchone()
                    s = result[0] if result and result[0] else 'UNKNOWN'
                    mycursor.close()
                    connection.close()
            except Error as e:
                print(f"Error: {e}")
                s = 'UNKNOWN'

            if confidence > 70:
                cv2.putText(img, s, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                try:
                    connection = get_db_connection()
                    if connection is not None:
                        mycursor = connection.cursor()
                        mycursor.execute("""INSERT INTO `accs_hist` 
                                        (`accs_date`, `accs_prsn`, `status`) 
                                        VALUES (CURDATE(), %s, 'Masuk')""", (s,))
                        connection.commit()
                        mycursor.close()
                        connection.close()
                except Exception as e:
                    print(f"Error saving to history: {e}")
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier(cascade_path)
    clf = cv2.face.LBPHFaceRecognizer_create()
    
    if not os.path.exists("classifier.xml"):
        return "Error: File classifier.xml tidak ditemukan! Silakan train classifier terlebih dahulu.", 500
        
    clf.read("classifier.xml")

    wCam, hCam = 500, 400

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return "Error: Tidak dapat mengakses kamera!", 500
        
    cap.set(3, wCam)
    cap.set(4, hCam)

    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break
                
            img = recognize(img, clf, faceCascade)

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
