from ._anvil_designer import Form1Template
from anvil import *
import anvil.server

class Form1(Form1Template):

  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run before the form opens.

  

  def End_click(self, **event_args):
        if (self.male.selected_value=="Male"):
            male=1
        else:
            male=0
        age=int(self.age.text)
        pregnancies=int(self.pregnants.text)
        if (self.educater.selected_value=="no highschool degree"):
            education=1.0
        elif(self.educater.selected_value=="highschool degree"):
            education=2.0
        elif(self.educater.selected_value=="Under grad"):
            education=3.0
        elif(self.educater.selected_value=="grad or higher"):
            education=4.0
        else:
            education=1.0
        if (self.Work_place_danger.selected_value=="No Danger"):
            work_type=1
        elif(self.Work_place_danger.selected_value=="Slight Danger"):
            work_type=2
        elif(self.Work_place_danger.selected_value=="Moderate Danger"):
            work_type=3
        elif(self.Work_place_danger.selected_value=="High Danger"):
            work_type=4
        else:
            work_type=2
        if (self.Current_Smoker.checked == True):
            currentSmoker=1
        else:
            currentSmoker=0
        cigsPerDay=float(self.Number_of_cigs.text)
        BPMeds=float(self.BPMED.text)
        if (self.hearter.checked == True):
            prevalentHeartIll=1
        else:
            prevalentHeartIll=0
        if (self.Stroke.checked == True):
            prevalentStroke=1
        else:
            prevalentStroke=0
        if (self.Past_Hypotension.checked == True):
            prevalentHyp=1
        else:
            prevalentHyp=0
        if (self.YellowFingers.checked == True):
            YellowFingers=1
        else:
            YellowFingers=0
        if (self.Anxiously.checked == True):
            Anxiety=1
        else:
            Anxiety=0
        if (self.Wheezer.checked == True):
            Wheezing=1
        else:
            Wheezing=0
        if (self.Alcohall.checked == True):
            Alcohol=1
        else:
            Alcohol=0
        if (self.Cougher.checked == True):
            Coughing=1
        else:
            Coughing=0
        
        if (self.Short_Breath.checked == True):
            ShortnessBreath=1
        else:
             ShortnessBreath=0
        if (self.Swallower.checked == True):
            Swallow=1
        else:
            Swallow=0
        if (self.Chest_hurter.checked == True):
            Chest_Hurts=1
        else:
            Chest_Hurts=0

        
        
       
        totChol= int(self.tot_chol.text)
        sysBP= int(self.SysBP.text)
        diaBP= int(self.diaBP.text)
        BMI=(float(self.weighter.text))/(((float(self.Heighter.text))*(float(self.Heighter.text)))/100)
        heartRate=int(self.heartRate.text)
        glucose=int(self.glucoser.text)
        blood_urea=int(self.Urea_level.text)
        haemoglobin=float(self.Haemoglobo.text)
        
        
        if (self.Anemia_katkam.checked == True):
            Anemia=1
        else:
            Anemia=0
        
        if (self.Appetizer.selected_value=="Good Appetite"):
            appetite=1
        elif(self.Appetizer.selected_value=="Bad Appetite"):
            appetite=0
            
            
        diabetes=anvil.server.call('diabetes_checker',pregnancies, glucose, diaBP, BMI, age)
        
        self.Diabeetus.visible=True
        if (diabetes==1) :
          self.Diabeetus.text="High chance of suffering from diabetes"
        else :
          self.Diabeetus.text="Low chance of suffering from diabetes"
        
        CAD_percentage=anvil.server.call('CAD_finder',male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)
       
        self.CAD_result.visible=True
        if CAD_percentage >= 25:
          self.CAD_result.text="The probability of you having CAD within the next 10 years is high (" + str(round(CAD_percentage, 2)) + "%)."
        elif CAD_percentage >= 0.01:
          self.CAD_result.text="The probability of you having CAD within the next 10 years is low (" + str(round(CAD_percentage, 2)) + "%)."
        else:
          self.CAD_result.text="There is minimal probability of you having CAD (<0.01%)"
        Stroked = anvil.server.call('Stroker', male, age, prevalentHyp, work_type, prevalentHeartIll, glucose, BMI)
        self.Strokered.visible=True
        if (Stroked==1):
          self.Strokered.text="High chance of suffering from Strokes"
        else :
          self.Strokered.text="Low chance of suffering from Strokes"
        Cancered= anvil.server.call('Vrishab',male, age, currentSmoker, YellowFingers, Anxiety, 1, Wheezing, Alcohol, Coughing, ShortnessBreath, Swallow, Chest_Hurts)
        self.Lung_cancer.visible=True
        if (Cancered==1):
          self.Lung_cancer.text="High chance of suffering from Lung Cancer"
        else :
          self.Lung_cancer.text="Low chance of suffering from Lung Cancer"
        if (CAD_percentage>38):
          CAD_percentage=1
        else:
          CAD_percentage=0
        KidneyPhail= anvil.server.call('Kidknees', age, diaBP, glucose, blood_urea, haemoglobin, CAD_percentage, appetite, Anemia)
        self.Kidsney_ok.visible=True;
        if (KidneyPhail==1):
            self.Kidsney_ok.text="High Chance of kidney failure"
        else:
            self.Kidsney_ok.text="Low Chance of kidney failure"
  pass



