If there is any question, please contact the author at haoyuansun@cuhk.edu.hk

# Abnormal gait trajectories prediction and gait phases recognition of children with cerebral palsy based on deep learning: A pilot study
## Abstract:
Cerebral palsy (CP) is a leading cause of physical disability in children, with an incidence rate of 0.14%-0.35%. Gait disturbance represents the primary symptom in children with CP, making it a treatment priority. As existing clinical and traditional rehabilitation methods face limitations, lower limb exoskeletons (LLEs) are increasingly utilized in rehabilitation treatment. Locomotion intention recognition algorithms are crucial for adapting LLEs to the diverse gait symptoms in children with CP; however, a locomotion intention recognition method specifically designed for this population remains a lack. This paper proposes a deep learning model to predict hip, knee, and ankle trajectories and recognize six gait phases in children with CP, which includes an abnormal gait feature extraction module, a deep optimized recursive prediction branch, and an imbalanced classification branch. Using public data, the model predicts hip, knee, and ankle trajectories and recognizes six gait phases with high accuracy in both intra- and inter-subject scenarios. The model achieves a Mean Absolute Error (MAE) of 0.04±0.01 and an R² of 0.95±0.04 in the intra-subject scenario, while in the inter-subject, the MAE is 0.04±0.01 and the R² is 0.96±0.02. For gait phase recognition, the model attains an accuracy of 93.42±0.62 and an F1 score of 93.44±0.62 in the intra-subject scenario, with both accuracy and F1 score at 92.75 for inter-subject. Comparative analyses with other models highlight the superior performance of our model, offering promising applications in recognizing abnormal gait locomotion intentions in children with CP.

## Keywords:
Cerebral palsy, Lower limb motion analysis, Imbalance classification, Gait trajectory, Gait phase

## Schematic diagram of the proposed method
<p align="center"> 
<img width="470" alt="image" src="https://github.com/user-attachments/assets/1aee4b11-cf0f-4f29-9a2d-103e209b37a3" />
</p>
