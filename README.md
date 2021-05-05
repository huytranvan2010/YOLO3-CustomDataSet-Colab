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



### Tài liệu tham khảo
