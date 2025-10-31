import joblib
import pandas as pd

def get_user_input():
    """
    Function to ask questions and collect user responses for mental health assessment
    """
    questions = {
        'Hours_Worked': "How many hours do you work per day? (Enter a number between 0-24): ",
        'Sleep_Hours': "How many hours do you sleep per day? (Enter a number between 0-24): ",
        'Stress_Level': "On a scale of 1-10, how would you rate your current stress level? (1=lowest, 10=highest): ",
        'Physical_Activity': "How many hours per week do you spend on physical activity? (Enter a number): ",
        'Social_Connection': "On a scale of 1-10, how would you rate your social connections? (1=very isolated, 10=very connected): ",
        'Work_Life_Balance': "On a scale of 1-10, how would you rate your work-life balance? (1=poor, 10=excellent): ",
        'Support_System': "On a scale of 1-10, how strong is your support system? (1=none, 10=very strong): ",
        'Job_Satisfaction': "On a scale of 1-10, how satisfied are you with your job? (1=very dissatisfied, 10=very satisfied): "
    }
    
    responses = {}
    print("\n=== Mental Health Assessment Questionnaire ===\n")
    
    try:
        for column, question in questions.items():
            while True:
                try:
                    response = float(input(question))
                    if column in ['Hours_Worked', 'Sleep_Hours']:
                        if 0 <= response <= 24:
                            break
                        print("Please enter a valid number between 0 and 24.")
                    elif 'scale' in question.lower():
                        if 1 <= response <= 10:
                            break
                        print("Please enter a valid number between 1 and 10.")
                    else:
                        if response >= 0:
                            break
                        print("Please enter a valid positive number.")
                except ValueError:
                    print("Please enter a valid number.")
            
            responses[column] = response
            print()  # Add a blank line between questions
        
    except KeyboardInterrupt:
        print("\nAssessment cancelled.")
        return None
    
    return responses

def assess_mental_health():
    """
    Main function to run the mental health assessment
    """
    print("Welcome to the Mental Health Assessment System")
    print("Please answer the following questions honestly for an accurate assessment.\n")
    
    try:
        # Load the trained model and label encoder
        model = joblib.load('mental_health_model.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        
        # Get user responses
        responses = get_user_input()
        
        if responses:
            print("\nAnalyzing your responses...")
            # Convert responses to DataFrame
            input_df = pd.DataFrame([responses])
            
            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Get the predicted class label
            predicted_class = label_encoder.inverse_transform(prediction)[0]
            
            # Get prediction probabilities
            class_probabilities = dict(zip(label_encoder.classes_, prediction_proba[0]))
            
            print("\n=== Mental Health Assessment Results ===")
            print(f"\nBased on your responses, you may be experiencing: {predicted_class}")
            print("\nProbability breakdown:")
            for state, prob in class_probabilities.items():
                print(f"{state}: {prob*100:.1f}%")
            
            # Provide some general recommendations
            print("\nGeneral Recommendations:")
            if float(responses['Sleep_Hours']) < 7:
                print("- Consider getting more sleep (7-9 hours is recommended)")
            if float(responses['Stress_Level']) > 7:
                print("- Your stress level is high. Consider stress management techniques")
            if float(responses['Physical_Activity']) < 3:
                print("- Try to increase your physical activity")
            if float(responses['Social_Connection']) < 5:
                print("- Consider strengthening your social connections")
            
            print("\nNote: This is a preliminary assessment. Please consult with a mental health professional for a proper diagnosis.")
    
    except FileNotFoundError:
        print("Error: Model files not found. Please make sure you have trained the model first.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    assess_mental_health()