# YOLO3-CustomDataSet-Colab

### Các bước để thực hiện Custom Object Detection
- **Bước 1**: Chuẩn bị dữ liệu
    * Download ảnh
    * Gán nhãn ảnh (khuyên dùng LabelImg)
    * Nén ảnh để vào `images.zip` để upload lên Drive
- **Bước 2**: Cài đặt Google Drive và Colab
    * Tạo folder `yolo3`
    * Upload `images.zip` vào folder `yolo3`
    * Tạo Notebook trong Colab
    * Mount Google Drive vào Colab
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
- **Bước 3**: Clone, configure, compile Darknet
    * Clone Darknet vào Colab
    ```python
    !git clone https://github.com/AlexeyAB/darknet
    ```
    * Configure Darknet - enable GPU và OpenCV
    Sửa trong Makefile thành GPU=1, CUDNN=1, OPENCV=1
    * Compile Darknet
    Hai phần trên có thể thực hiện được bằng các câu lệnh sau
    ```python
    # Chuyển đến thư mục darknet
    %cd darknet
    !sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    !sed -i 's/GPU=0/GPU=1/' Makefile
    !sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    !make
    # câu lệnh sed -i 's/OPENCV=0/OPENCV=1/' Makfile để thay OPENCV=0 thành OPENCV=1 trong Makefile (sed - stream editing, -i thực hiện trên original file luôn)
    ```
- **Bước 4**: Configue .cfg file (nằm trong dark/cfg). Trong này có sẵn cả yolov3.cfg và yolo3-tiny.cfg (nhanh hơn nhưng accuracy thấp hơn, tùy mục đích sử dụng)

    - **4.1a.** Đối với `yolo3-tiny.cfg`
        - Tạo một bản copy của `yolo3-tiny.cfg` (good practice)
        ```python
        !cp cfg/yolo3-tiny.cfg cfg/yolov3-tiny_training.cfg
        ```
        - Thay đổi một số dòng trong `yolov3-tiny_training.cfg`
            - Thay đổi dòng batch thành `batch=64`
            - Thay đổi dòng subdivisions thành `subdivisions=16`
            - Thay đổi dòng max_batches thành `classes*2000`, nếu 1 class thì `max_batches=2000`, nếu 2 classes thì `max_batches=4000`, nếu 3 classes thì `max_batches=6000`
            - Dòng `L127 và L171` thay đổi filters=255 thành filters=(classes + 5)*3. Nếu classes=1 thì filters=18, classes=2 thì filters=21, classes=3 thì filters=24
            - Dòng `L135 và L177` thay đổi classes=80 thành classes=no. of objects. Nếu no. of objects=1 thì classes=1, no. of objects=2 thì classes=2, no. of objects=3 thì classes=3

    - **4.1b.** Đối với `yolov3.cfg`
        - Tạo một bản copy của `yolov3.cfg`
        ```python
        !cp cfg/yolo3-tiny.cfg cfg/yolov3_training.cfg
        ```
        - Thay đổi một số dòng trong `yolov3_training.cfg`
            - Thay đổi dòng batch thành `batch=64`
            - Thay đổi dòng subdivisions thành `subdivisions=16`
            - Thay đổi dòng max_batches thành `classes*2000`, nếu 1 class thì `max_batches=2000`, nếu 2 classes thì `max_batches=4000`, nếu 3 classes thì `max_batches=6000`
            - Dòng `L603, L689 và L776` thay đổi filters=255 thành filters=(classes + 5)*3. Nếu classes=1 thì filters=18, classes=2 thì filters=21, classes=3 thì filters=24
            - Dòng `L610, L696 và L783` thay đổi classes=80 thành classes=no. of objects. Nếu no. of objects=1 thì classes=1, no. of objects=2 thì classes=2, no. of objects=3 thì classes=3 
        ```python
        !sed -i 's/batch=1/batch=64/' cfg/yolov3_training.cfg
        !sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_training.cfg
        !sed -i 's/max_batches = 500200/max_batches = 6000/' cfg/yolov3_training.cfg
        !sed -i '610 s@classes=80@classes=3@' cfg/yolov3_training.cfg
        !sed -i '696 s@classes=80@classes=3@' cfg/yolov3_training.cfg
        !sed -i '783 s@classes=80@classes=3@' cfg/yolov3_training.cfg
        !sed -i '603 s@filters=255@filters=24@' cfg/yolov3_training.cfg
        !sed -i '689 s@filters=255@filters=24@' cfg/yolov3_training.cfg
        !sed -i '776 s@filters=255@filters=24@' cfg/yolov3_training.cfg
        ```

    - **4.1c.** Checking
        - Vào các file vừa tạo ra nhấn `Ctrl + F` và nhập `[yolo]`
            - Đối với yolov3-tiny.cfg cần `2` chỗ khớp
            - Đối với yolov3.cfg cần `3` chỗ khớp
            - Kiểm tra `filters` trước `[yolo]` và `classes` phía sau `[yolo]`
- **Bước 5**: Tạo file `.names` trong thư mục `darknet/data`
    * Tạo file `obj.names` để chứa các labels, làm xong nên kiểm tra lại
    ```python
    !echo -e 'Wearing Mask\n2nd item\n3rd item' > data/obj.names
    ```
- **Bước 6**: Tạo file `.data` trong thư mục `darknet/data`
    * Tạo file `obj.data`. File này chứa 5 dòng. Làm xong cũng nên kiểm tra
        1. Số lượng objects
        2. Path dẫn đến file `train.txt`
        3. Path dẫn đến file `test.txt`
        4. Path dẫn đến file `obj.names`
        5. Path dẫn đến `trained yolo weights`
    ```python
    !echo -e 'classes= 3\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = /mydrive/yolov3' > data/obj.data
    ```
- **Bước 7**: Tạo folder cho các ảnh
    * Tạo folder `obj` trong folder `darknet/data` (hiện tại thư mục này vừa chứa ảnh vừa chứa file .txt)
    ```python
    !mkdir data/obj
    ```
    * Giải nén `images.zip` từ thư mục `yolov3` trong Google drive sang thư mục `data/obj`
    ```python
    !unzip /mydrive/yolov3/images.zip -d data/obj
    ```
- **Bước 8**: Tạo file `train.txt`
    * Tạo file `train.txt` trong thư mục `darknet/data`
        - Mỗi file được lưu ở 1 dòng
        - Đường dẫn đến file là relative so với thư mục `darknet`
    ```python
    import glob
    images_list = glob.glob("data/obj/*.jpg")
    with open("data/train.txt", "w") as f:
    f.write("\n".join(images_list))
    ```

### Tài liệu tham khảo
https://www.youtube.com/watch?v=DLngCtsG3bk

https://github.com/emasterclassacademy/Single-Multiple-Custom-Object-Detection

https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects