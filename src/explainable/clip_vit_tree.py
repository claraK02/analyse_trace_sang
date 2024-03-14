


from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import numpy as np


import numpy as np

data_list = [
    "Altération glissée",
    "Trace à l'intérieur d'un canon d'arme à feu",
    "sang aspiré",
    "Trace jaunâtre accolé à une trace rougeâtre",
    "Trace de sérum",
    "Trace à la couleur peu prononcée et/ou couleur prononcée uniquement sur le contour",
    "Altération diluée",
    "Zone d'absence de traces dessinée par des traces en périphérie partielle ou totale",
    "Zone d'interruption",
    "Trace centimétrique",
    "Association à des traces millimétrique",
    "Association à des traces centimétrique",
    "Variation continue des diamètres du cm au mm",
    "Forme circulaire",
    "Variationté continue des diamètres du cm au mm",
    "Présence de points blancs",
    "Traces millimétriques majoritairement ovoïdes",
    "Distribution linéaire",
    "Sang expiré",
    "Volume impacté/Sang propulsé (surface secondaire)",
    "Trace millimétriques très étirées",
    "Altération du contour et/ou dépression centrale et/ou extension linéaire",
    "Forme ovoïde",
    "Présence de points blancs",
    "Trace millimétriques majoritairement ovoïdes",
    "Répartition homogène des traces millimétriques autour de la trace centimétrique",
    "Projection",
    "Trace d'insecte",
    "Distribution convergente",
    "Cheminement",
    "Trace passive (surface horizontale)",
    "Trace passive (surface non horizontale)",
    "Forme selon le support",
    "Sang expiré",
    "Volume impacté/Sang propulsé (surface secondaire)",
    "Traces millimétriques très étirées",
    "Volume impacté (surface primaire)",
    "Trace centimétrique très épineuse",
    "Goutte à goutte",
    "Trace passive (surface horizontale) trace d'accompagnement",
    "Répartition homogène des traces millimétriques autour de la trace centimétrique",
    "Modèle d'impact",
    "Distribution linéaire",
    "Surface étudiée poreuse",
    "Forme en bande",
    "Sang propulsé (surface primaire)",
    "Chute de volume",
    "Imprégnation",
    "Volume impacté (surface primaire)",
    "Trace centimétrique très épineuse",
    "Accumulation",
    "Coulée",
    "Mécanismes transférants",
    "Goutte à goutte",
    "Trace d'accompagnement Trace passive (surface horizontale)",
    "Distribution parabolique descendante",
    "Localisée à la convergence de traces millimétriques ovoïdes",
    "Existence d'une striation interne",
    "Sang propulsé (surface primaire)",
    "Chute de volume",
    "Projection gravitationnelle",
    "Modèle d'éjection",
    "Foyer de modèle d'impact",
    "Distribution dense",
    "Trace préexistante",
    "Sang majoritairement  dans la forme",
    "Sang vaporisé",
    "Forme ovoïde",
    "Transfert glissé",
    "Transfert par contact",
    "Altération par contact",
    "Projection ovoïde",
    "Projection circulaire"
]



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

    return class_probs

if __name__ == "__main__":

    #PATH=r"C:\Users\Yanis\Documents\Cours Centrale Marseille\Projet 3A\data\data_labo\test_256\3- Modèle Transfert par contact\lino\461.jpg"
    PATH=r"C:\Users\Yanis\Documents\Cours Centrale Marseille\Projet 3A\data\data_labo\train_256\11- Modèle d'éjection\lino\3674.jpg"
    tree = np.array([
    ["centimetric stain associated with some millimetric stains ", "centimetric stain associated with others centimetrics stains "],
    ["continuous variation of diameters from cm to mm", "alteration of contour or central depression or linear extension"],
    ["descending parabolic distribution", "located at the convergence of ovoid millimetric traces"],
    ["dense distribution", "impact model focal point"],
    ["milimetric stains mainly ovoid", "presence of white points"],
    ["milimetric stains very slandered", "Homogeneous distribution of milimetric stains around a centimetric stain"],
    ["presence of spiked centimetric stains", "no presence of a centimetric stain"],
    ["shape of the stain impacted by the support", "shape of the stain not impacted by the support"],
    ["porous surface of support", "striped shaped stain"],
    ["internal striations", "no internal striations"],
    ["preexisting stains", "blood mainly in the shape"],
    ["dense distribution", "not dense distribution"],
    ["ovoid shape", "not ovoid shape"]

])

    


    open_image = Image.open(PATH)
    image = np.array(open_image)


    for k in tree:
        print(classify_image_with_vit(image, list(k)))
        print("\n")
