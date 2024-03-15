


from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import numpy as np


import numpy as np

class_french = [
    "Trace jaunâtre accolé à une trace rougeâtre",
    "Trace à la couleur peu prononcée et/ou couleur prononcée uniquement sur le contour",
    "Zone d'absence de traces dessinée par des traces en périphérie partielle ou totale",
    "Trace centimétrique",
    "Association à des traces millimétrique",
    "Variation continue des diamètres du cm au mm",
    "Forme circulaire",
    "Présence de points blancs",
    "Traces millimétriques majoritairement ovoïdes",
    "Distribution linéaire",
    "Altération du contour et/ou dépression centrale et/ou extension linéaire",
    "Répartition homogène des traces millimétriques autour de la trace centimétrique",
    "Trace d'insecte",
    "Distribution convergente",
    "Surface horizontale",
    "Trace centimétrique très épineuse",
    "Surface étudiée poreuse",
    "Forme en bande",
    "Coulée",
    "Distribution parabolique descendante",
    "Localisée à la convergence de traces millimétriques ovoïdes",
    "Existence d'une striation interne",
    "Distribution dense",
    "Sang majoritairement  dans la forme",
    "Sang vaporisé",
]

class_english = [
    "Yellowish trace attached to a reddish trace",
    "Trace with poorly pronounced color and/or color pronounced only on the outline",
    "Area without traces drawn by partial or total peripheral traces",
    "Centimeter-sized trace",
    "Association with millimeter-sized traces",
    "Continuous variation of diameters from cm to mm",
    "Circular shape",
    "Presence of white dots",
    "Mainly ovoid millimeter-sized traces",
    "Linear distribution",
    "Alteration of the contour and/or central depression and/or linear extension",
    "Homogeneous distribution of millimeter-sized traces around the centimeter-sized trace",
    "Insect trace",
    "Convergent distribution",
    "Horizontal surface",
    "Very thorny centimeter-sized trace",
    "Porous studied surface",
    "Band shape",
    "Flow",
    "Descending parabolic distribution",
    "Located at the convergence of ovoid millimeter-sized traces",
    "Existence of an internal striation",
    "Dense distribution",
    "Blood mainly in the shape",
    "Vaporized blood",
]

binary_list = [
    ["Yellowish trace attached to a reddish trace", "Trace not attached to a reddish trace"],
    ["Trace with poorly pronounced color and/or color pronounced only on the outline", "Trace with pronounced color all over"],
    ["Area without traces drawn by partial or total peripheral traces", "Area with traces drawn by partial or total peripheral traces"],
    ["Centimeter-sized trace", "Trace not centimeter-sized"],
    ["Association with millimeter-sized traces", "No association with millimeter-sized traces"],
    ["Continuous variation of diameters from cm to mm", "No continuous variation of diameters from cm to mm"],
    ["Circular shape", "Non-circular shape"],
    ["Presence of white dots", "Absence of white dots"],
    ["Mainly ovoid millimeter-sized traces", "Mainly non-ovoid millimeter-sized traces"],
    ["Linear distribution", "Non-linear distribution"],
    ["Alteration of the contour and/or central depression and/or linear extension", "No alteration of the contour and/or central depression and/or linear extension"],
    ["Homogeneous distribution of millimeter-sized traces around the centimeter-sized trace", "Non-homogeneous distribution of millimeter-sized traces around the centimeter-sized trace"],
    ["Insect trace", "Non-insect trace"],
    ["Convergent distribution", "Non-convergent distribution"],
    ["Horizontal surface", "Non-horizontal surface"],
    ["Very thorny centimeter-sized trace", "Centimeter-sized trace not very thorny"],
    ["Porous studied surface", "Non-porous studied surface"],
    ["Band shape", "Non-band shape"],
    ["Flow", "No flow"],
    ["Descending parabolic distribution", "Non-descending parabolic distribution"],
    ["Located at the convergence of ovoid millimeter-sized traces", "Not located at the convergence of ovoid millimeter-sized traces"],
    ["Existence of an internal striation", "No existence of an internal striation"],
    ["Dense distribution", "Non-dense distribution"],
    ["Blood mainly in the shape", "Blood not mainly in the shape"],
    ["Vaporized blood", "Non-vaporized blood"],
]


def get_criterions(np_image):
    binary_list = [
    ["Yellowish blood stain attached to a reddish blood stain", "Blood stain not attached to a reddish blood stain"],
    ["Blood stain with poorly pronounced color and/or color pronounced only on the outline", "Blood stain with pronounced color all over"],
    ["Area without blood stains drawn by partial or total peripheral blood stains", "Area with blood stains drawn by partial or total peripheral blood stains"],
    ["Centimeter-sized blood stain", "Blood stain not centimeter-sized"],
    ["Association with millimeter-sized blood stains", "No association with millimeter-sized blood stains"],
    ["Continuous variation of diameters from cm to mm of blood stains", "No continuous variation of diameters from cm to mm of blood stains"],
    ["Circular shape of blood stain", "Non-circular shape of blood stain"],
    ["Ovoid shape of blood stain", "Non-ovoid shape of blood stain"],
    ["Presence of white dots in blood stain", "Absence of white dots in blood stain"],
    ["Mainly ovoid millimeter-sized blood stains", "Mainly non-ovoid millimeter-sized blood stains"],
    ["Linear distribution of blood stains", "Non-linear distribution of blood stains"],
    ["Alteration of the contour and/or central depression and/or linear extension of blood stain", "No alteration of the contour and/or central depression and/or linear extension of blood stain"],
    ["Homogeneous distribution of millimeter-sized blood stains around the centimeter-sized blood stain", "Non-homogeneous distribution of millimeter-sized blood stains around the centimeter-sized blood stain"],
    ["Insect blood stain", "Non-insect blood stain"],
    ["stripe-shaped blood stain", "Non-stripe-shaped blood stain"],
    ["Convergent distribution of blood stains", "Non-convergent distribution of blood stains"],
    ["Horizontal surface of blood stain", "Non-horizontal surface of blood stain"],
    ["Very thorny centimeter-sized blood stain", "Centimeter-sized blood stain not very thorny"],
    ["Porous studied surface of blood stain", "Non-porous studied surface of blood stain"],
    ["Descending parabolic distribution of blood stains", "Non-descending parabolic distribution of blood stains"],
    ["Located at the convergence of ovoid millimeter-sized blood stains", "Not located at the convergence of ovoid millimeter-sized blood stains"],
    ["Existence of an internal striation in blood stain", "No existence of an internal striation in blood stain"],
    ["Dense distribution of blood stains", "Non-dense distribution of blood stains"],
    ["Blood mainly in the shape of blood stain", "Blood not mainly in the shape of blood stain"],
    ["Vaporized blood stain", "Non-vaporized blood stain"],
]
    
    list_criteria = []
    for k in binary_list:
        result = classify_image_with_vit(np_image, list(k))
        if result == 0:
            list_criteria.append(k[0])
        
    
    return list_criteria




def classify_image_with_vit(np_image, class_choices):
    # Load the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Convert the numpy image to a PIL image
    image = Image.fromarray(np_image)

    # Prepare the inputs
    inputs = processor(text=class_choices, images=image, return_tensors="pt", padding=True)

    # Get the model outputs
    outputs = model(**inputs)

    # Get the image-text similarity score
    logits_per_image = outputs.logits_per_image

    # Get the label probabilities
    probs = logits_per_image.softmax(dim=1)

    # Store the probability for each class
    class_probs = {class_choices[i]: prob.item() for i, prob in enumerate(probs[0])}

    # Print the probabilities for each class
    for class_choice, prob in class_probs.items():
        print(f"Probability of '{class_choice}': {prob}")

    # if the first element of binary_list has the highest probability, return 0 else return 1
    if class_probs[class_choices[0]] > class_probs[class_choices[1]]:
        return 0
    else:
        return 1

if __name__ == "__main__":

    #PATH=r"C:\Users\Yanis\Documents\Cours Centrale Marseille\Projet 3A\data\data_labo\test_256\3- Modèle Transfert par contact\lino\461.jpg"
    PATH=r"C:\Users\Yanis\Documents\Cours Centrale Marseille\Projet 3A\data\data_labo\test_256\1- Modèle Traces passives\carrelage\100.jpg"

    # Load the image
    image = Image.open(PATH)
    np_image = np.array(image)

    # Get the criterions
    criterions = get_criterions(np_image)
    print(criterions)
