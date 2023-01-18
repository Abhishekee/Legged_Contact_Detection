# Legged_Contact_Detection
Developed contact estimation of humanoid robots using supervised learning-based neural network 
architecture. Force, torque, acceleration, and angular velocity in the x, y, and z axis are given as input.
Implemented Artificial Neural Network (ANN) to predict probabilities for the robot's stable contact and 
unstable contact. This architecture can also be used for contact predictions of point feet robots.
he proposed model achieved stable accuracy of 99.5% and slip accuracy of 98.5% with an AUC of 99%.
The model was also tested on unseen datasets with a maximum AUC of 98.2%


For training the model enter code given below:

```
python3  trainer.py
```



Given below are the evaluation metrics on test and unseen data

a)Confusion Marix on Test Data (ATLAS_21K)----------                                      b)Confusion Matrix on Unseen Data (ATLAS_7K)

![Confm_ATLAS_21K](https://user-images.githubusercontent.com/111289395/211140436-b2ee1f67-eff1-4081-95c1-9d34085d09d5.png)               ![Confm_ATLAS_7K](https://user-images.githubusercontent.com/111289395/211140526-26f54557-5fdf-4b71-ae7c-7d4eac3bcd39.png)
           
a)Confusion Marix on Unseen Data (ATLAS_10K)----------                                      b)Confusion Matrix on Unseen Data (ATLAS_50K)

![Confm_ATLAS_10K](https://user-images.githubusercontent.com/111289395/211140659-c5c479df-2ba0-451c-8b42-afdd31c7e4e2.png)              ![Confm_ATLAS_50K](https://user-images.githubusercontent.com/111289395/211140673-98eea52a-ec0b-40d6-94cd-93ad9626db08.png)

a) Precision-Recall curve on the Test Data (ATLAS_21K)-                         b) Precision-Recall curve on Unseen Data(ATLAS_7K) 
                           
![PR_ATLAS_21K](https://user-images.githubusercontent.com/111289395/211140959-ff04e751-da75-45a6-be49-2b93fe26588e.png)                  ![PR_ATLAS_7K](https://user-images.githubusercontent.com/111289395/211140967-9a2d20b2-692d-420d-8e1a-46f4db4a6ab6.png)

a) Precision-Recall curve on the Unseen Data (ATLAS_10K)-                        b) Precision-Recall curve on Unseen Data(ATLAS_50K)  

![PR_ATLAS_10K](https://user-images.githubusercontent.com/111289395/211140986-428fb62c-99b9-49c9-98d6-7ff51aed3383.png)                  ![PR_ATLAS_50K](https://user-images.githubusercontent.com/111289395/211141003-0effe74a-d1dd-4bf3-b79e-c6d99f98d0f8.png)



 


