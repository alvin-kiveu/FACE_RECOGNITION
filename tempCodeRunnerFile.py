timestamp = int(time.time())
                image_path = "captured_image_{}.jpg".format(timestamp)
                cv2.imwrite(image_path, detected_frame)
                print("Captured image saved as:", image_path)