import gradio as gr
import custom.OneButtonPrompt.shared
from custom.OneButtonPrompt.shared import add_ctrl

from random_prompt.build_dynamic_prompt import build_dynamic_prompt, OBPresets

from random_prompt.csv_reader import load_config_csv

from .modules.settings import default_settings
from random_prompt.one_button_presets import OneButtonPresets

#OBPresets = OneButtonPresets()
settings = default_settings
custom_obp_values = OBPresets.get_obp_preset(settings["OBP_preset"])


insanitylevel = 5
subjects = ["all"]
subjectsubtypesobject = ["all"]
subjectsubtypeshumanoid = ["all"]
subjectsubtypesconcept = ["all"]
modeltypelist = ["SDXL"] #, "Anime Model"]
artists = [
    "all",
    "all (wild)",
    "none",
    "popular",
    "greg mode",
    "3D",
    "abstract",
    "angular",
    "anime",
    "architecture",
    "art nouveau",
    "art deco",
    "baroque",
    "bauhaus",
    "cartoon",
    "character",
    "children's illustration",
    "cityscape",
    "clean",
    "cloudscape",
    "collage",
    "colorful",
    "comics",
    "cubism",
    "dark",
    "detailed",
    "digital",
    "expressionism",
    "fantasy",
    "fashion",
    "fauvism",
    "figurativism",
    "gore",
    "graffiti",
    "graphic design",
    "high contrast",
    "horror",
    "impressionism",
    "installation",
    "landscape",
    "light",
    "line drawing",
    "low contrast",
    "luminism",
    "magical realism",
    "manga",
    "melanin",
    "messy",
    "monochromatic",
    "nature",
    "nudity",
    "photography",
    "pop art",
    "portrait",
    "primitivism",
    "psychedelic",
    "realism",
    "renaissance",
    "romanticism",
    "scene",
    "sci-fi",
    "sculpture",
    "seascape",
    "space",
    "stained glass",
    "still life",
    "storybook realism",
    "street art",
    "streetscape",
    "surrealism",
    "symbolism",
    "textile",
    "ukiyo-e",
    "vibrant",
    "watercolor",
    "whimsical",
]
imagetypes = [
    "all",
    "all - force multiple",
    "all - anime",
    "none",
    "photograph",
    "octane render",
    "digital art",
    "concept art",
    "painting",
    "portrait",
    "anime",
    "only other types",
    "only templates mode",
    "dynamic templates mode",
    "art blaster mode",
    "quality vomit mode",
    "color cannon mode",
    "unique art mode",
    "massive madness mode",
    "photo fantasy mode",
    "subject only mode",
    "fixed styles mode",
    "the tokinator",
]
promptmode = ["at the back", "in the front"]
promptcompounder = ["1", "2", "3", "4", "5"]
ANDtogglemode = [
    "none",
    "automatic",
    "prefix AND prompt + suffix",
    "prefix + prefix + prompt + suffix",
]
seperatorlist = ["comma", "AND", "BREAK"]
genders = ["all", "male", "female"]

qualitymodelist = ["highest", "gated"]
qualitykeeplist = ["keep used", "keep all"]

promptenhancelist = ["none", "hyperprompt"]

generatevehicle = True
generateobject = True
generatefood = True
generatebuilding = True
generatespace = True
generateflora = True

generateanimal = True
generatebird = True
generatecat = True
generatedog = True
generateinsect = True
generatepokemon = True
generatemarinelife = True

generatemanwoman = True
generatemanwomanrelation = True
generatemanwomanmultiple = True
generatefictionalcharacter = True
generatenonfictionalcharacter = True
generatehumanoids = True
generatejob = True
generatefirstnames = True

generatelandscape = True
generatelocation = True
generatelocationfantasy = True
generatelocationscifi = True
generatelocationvideogame = True
generatelocationbiome = True
generatelocationcity = True

generateevent = True
generateconcepts = True
generatepoemline = True
generatesongline = True
generatecardname = True
generateepisodetitle = True
generateconceptmixer = True

config = load_config_csv()

for item in config:
    # objects
    if item[0] == "subject_vehicle" and item[1] != "on":
        generatevehicle = False
    if item[0] == "subject_object" and item[1] != "on":
        generateobject = False
    if item[0] == "subject_food" and item[1] != "on":
        generatefood = False
    if item[0] == "subject_building" and item[1] != "on":
        generatebuilding = False
    if item[0] == "subject_space" and item[1] != "on":
        generatespace = False
    if item[0] == "subject_flora" and item[1] != "on":
        generateflora = False
    # animals
    if item[0] == 'subject_animal' and item[1] != 'on':
        generateanimal = False
    if item[0] == 'subject_bird' and item[1] != 'on':
        generatebird = False
    if item[0] == 'subject_cat' and item[1] != 'on':
        generatecat = False
    if item[0] == 'subject_dog' and item[1] != 'on':
        generatedog = False
    if item[0] == 'subject_insect' and item[1] != 'on':
        generateinsect = False
    if item[0] == 'subject_pokemon' and item[1] != 'on':
        generatepokemon = False
    if item[0] == 'subject_marinelife' and item[1] != 'on':
        generatemarinelife = False
    # humanoids
    if item[0] == "subject_manwoman" and item[1] != "on":
        generatemanwoman = False
    if item[0] == "subject_manwomanrelation" and item[1] != "on":
        generatemanwomanrelation = False
    if item[0] == "subject_manwomanmultiple" and item[1] != "on":
        generatemanwomanmultiple = False
    if item[0] == "subject_fictional" and item[1] != "on":
        generatefictionalcharacter = False
    if item[0] == "subject_nonfictional" and item[1] != "on":
        generatenonfictionalcharacter = False
    if item[0] == "subject_humanoid" and item[1] != "on":
        generatehumanoids = False
    if item[0] == "subject_job" and item[1] != "on":
        generatejob = False
    if item[0] == "subject_firstnames" and item[1] != "on":
        generatefirstnames = False
    # landscape
    if item[0] == 'subject_location' and item[1] != 'on':
        generatelocation = False
    if item[0] == 'subject_location_fantasy' and item[1] != 'on':
        generatelocationfantasy = False
    if item[0] == 'subject_location_scifi' and item[1] != 'on':
        generatelocationscifi = False
    if item[0] == 'subject_location_videogame' and item[1] != 'on':
        generatelocationvideogame = False
    if item[0] == 'subject_location_biome' and item[1] != 'on':
        generatelocationbiome = False
    if item[0] == 'subject_location_city' and item[1] != 'on':
        generatelocationcity = False
    # concept
    if item[0] == "subject_event" and item[1] != "on":
        generateevent = False
    if item[0] == "subject_concept" and item[1] != "on":
        generateconcepts = False
    if item[0] == "poemline" and item[1] != "on":
        generatepoemline = False
    if item[0] == "songline" and item[1] != "on":
        generatesongline = False
    if item[0] == "subject_cardname" and item[1] != "on":
        generatecardname = False
    if item[0] == "subject_episodetitle" and item[1] != "on":
        generateepisodetitle = False
    if item[0] == "subject_conceptmixer" and item[1] != "on":
        generateconceptmixer = False

# build up all subjects we can choose based on the loaded config file
if(generatevehicle or generateobject or generatefood or generatebuilding or generatespace or generateflora):
    subjects.append("--- object - all")
    if(generateobject):
          subjects.append("object - generic")
    if(generatevehicle):
          subjects.append("object - vehicle")
    if(generatefood):
          subjects.append("object - food")
    if(generatebuilding):
          subjects.append("object - building")
    if(generatespace):
          subjects.append("object - space")
    if(generateflora):
          subjects.append("object - flora")
          
if(generateanimal or generatebird or generatecat or generatedog or generateinsect or generatepokemon or generatemarinelife):
    subjects.append("--- animal - all")
    if(generateanimal):
        subjects.append("animal - generic")
    if(generatebird):
        subjects.append("animal - bird")
    if(generatecat):
        subjects.append("animal - cat")
    if(generatedog):
        subjects.append("animal - dog")
    if(generateinsect):
        subjects.append("animal - insect")
    if(generatemarinelife):
        subjects.append("animal - marine life")
    if(generatepokemon):
        subjects.append("animal - pokémon")

if(generatemanwoman or generatemanwomanrelation or generatefictionalcharacter or generatenonfictionalcharacter or generatehumanoids or generatejob or generatemanwomanmultiple):
    subjects.append("--- human - all")
    if(generatemanwoman):
        subjects.append("human - generic")
    if(generatemanwomanrelation):
        subjects.append("human - relations")
    if(generatenonfictionalcharacter):
        subjects.append("human - celebrity")
    if(generatefictionalcharacter):
        subjects.append("human - fictional")
    if(generatehumanoids):
        subjects.append("human - humanoids")
    if(generatejob):
        subjects.append("human - job/title")
    if(generatefirstnames):
        subjects.append("human - first name")
    if(generatemanwomanmultiple):
        subjects.append("human - multiple")

if(generatelandscape or generatelocation or generatelocationfantasy or generatelocationscifi or generatelocationvideogame or generatelocationbiome or generatelocationcity):
    subjects.append("--- landscape - all")
    if(generatelocation):
        subjects.append("landscape - generic")
    if(generatelocationfantasy):
        subjects.append("landscape - fantasy")
    if(generatelocationscifi):
        subjects.append("landscape - sci-fi")
    if(generatelocationvideogame):
        subjects.append("landscape - videogame")
    if(generatelocationbiome):
        subjects.append("landscape - biome")
    if(generatelocationcity):
        subjects.append("landscape - city")

if(generateevent or generateconcepts or generatepoemline or generatesongline or generatecardname or generateepisodetitle or generateconceptmixer):
    subjects.append("--- concept - all")
    if(generateevent):
        subjects.append("concept - event")
    if(generateconcepts):
        subjects.append("concept - the x of y")
    if(generatepoemline):
        subjects.append("concept - poem lines")
    if(generatesongline):
        subjects.append("concept - song lines")
    if(generatecardname):
        subjects.append("concept - card names")
    if(generateepisodetitle):
        subjects.append("concept - episode titles")
    if(generateconceptmixer):
        subjects.append("concept - mixer")


# do the same for the subtype subjects
# subjectsubtypesobject = ["all"]
# subjectsubtypeshumanoid = ["all"]
# subjectsubtypesconcept = ["all"]

# objects first
if generateobject:
    subjectsubtypesobject.append("generic objects")
if generatevehicle:
    subjectsubtypesobject.append("vehicles")
if generatefood:
    subjectsubtypesobject.append("food")
if generatebuilding:
    subjectsubtypesobject.append("buildings")
if generatespace:
    subjectsubtypesobject.append("space")
if generateflora:
    subjectsubtypesobject.append("flora")

# humanoids (should I review descriptions??)
if generatemanwoman:
    subjectsubtypeshumanoid.append("generic humans")
if generatemanwomanrelation:
    subjectsubtypeshumanoid.append("generic human relations")
if generatenonfictionalcharacter:
    subjectsubtypeshumanoid.append("celebrities e.a.")
if generatefictionalcharacter:
    subjectsubtypeshumanoid.append("fictional characters")
if generatehumanoids:
    subjectsubtypeshumanoid.append("humanoids")
if generatejob:
    subjectsubtypeshumanoid.append("based on job or title")
if generatefirstnames:
    subjectsubtypeshumanoid.append("based on first name")
if generatemanwomanmultiple:
    subjectsubtypeshumanoid.append("multiple humans")

# concepts
if generateevent:
    subjectsubtypesconcept.append("event")
if generateconcepts:
    subjectsubtypesconcept.append("the X of Y concepts")
if generatepoemline:
    subjectsubtypesconcept.append("lines from poems")
if generatesongline:
    subjectsubtypesconcept.append("lines from songs")
if generatecardname:
    subjectsubtypesconcept.append("names from card based games")
if generateepisodetitle:
    subjectsubtypesconcept.append("episode titles from tv shows")
if generateconceptmixer:
    subjectsubtypesconcept.append("concept mixer")


def ui_onebutton(prompt, run_event, random_button):
    def gen_prompt(
        insanitylevel,
        subject,
        artist,
        imagetype,
        antistring,
        prefixprompt,
        suffixprompt,
        givensubject,
        smartsubject,
        giventypeofimage,
        imagemodechance,
        chosengender,
        chosensubjectsubtypeobject,
        chosensubjectsubtypehumanoid,
        chosensubjectsubtypeconcept,
        givenoutfit,
        OBP_preset,
        promptenhance,
        modeltype,
    ):
        prompt = build_dynamic_prompt(
            insanitylevel,
            subject,
            artist,
            imagetype,
            False,
            antistring,
            prefixprompt,
            suffixprompt,
            1,
            "comma",
            givensubject,
            smartsubject,
            giventypeofimage,
            imagemodechance,
            chosengender,
            chosensubjectsubtypeobject,
            chosensubjectsubtypehumanoid,
            chosensubjectsubtypeconcept,
            False,
            False,
            0,
            givenoutfit,
            False,
            modeltype,
            OBP_preset,
            promptenhance,
        )

        return prompt 

    def instant_gen_prompt(
        insanitylevel,
        subject,
        artist,
        imagetype,
        antistring,
        prefixprompt,
        suffixprompt,
        givensubject,
        smartsubject,
        giventypeofimage,
        imagemodechance,
        chosengender,
        chosensubjectsubtypeobject,
        chosensubjectsubtypehumanoid,
        chosensubjectsubtypeconcept,
        givenoutfit,
        OBP_preset,
        promptenhance,
        modeltype,
        run_event,
    ):
        prompt = build_dynamic_prompt(
            insanitylevel,
            subject,
            artist,
            imagetype,
            False,
            antistring,
            prefixprompt,
            suffixprompt,
            1,
            "comma",
            givensubject,
            smartsubject,
            giventypeofimage,
            imagemodechance,
            chosengender,
            chosensubjectsubtypeobject,
            chosensubjectsubtypehumanoid,
            chosensubjectsubtypeconcept,
            False,
            False,
            0,
            givenoutfit,
            False,
            modeltype,
            OBP_preset,
            promptenhance, 
        )

        return prompt, run_event+1

    def add_prompt(
        prompt,
        insanitylevel,
        subject,
        artist,
        imagetype,
        antistring,
        prefixprompt,
        suffixprompt,
        givensubject,
        smartsubject,
        giventypeofimage,
        imagemodechance,
        chosengender,
        chosensubjectsubtypeobject,
        chosensubjectsubtypehumanoid,
        chosensubjectsubtypeconcept,
        givenoutfit,
        OBP_preset,
        promptenhance,
        modeltype,
    ):
        prompt = (
            prompt
            + "---"
            + build_dynamic_prompt(
                insanitylevel,
                subject,
                artist,
                imagetype,
                False,
                antistring,
                prefixprompt,
                suffixprompt,
                1,
                "comma",
                givensubject,
                smartsubject,
                giventypeofimage,
                imagemodechance,
                chosengender,
                chosensubjectsubtypeobject,
                chosensubjectsubtypehumanoid,
                chosensubjectsubtypeconcept,
                False,
                False,
                0,
                givenoutfit,
                False,
                modeltype,
                OBP_preset,
                promptenhance,
            )
        )

        return prompt

    with gr.Tab(label="OneButtonPrompt"):
        with gr.Row():
            #instant_obp = gr.Button(value="Instant OBP", size="sm", min_width = 1)
            #random_button = gr.Button(value="Random Prompt", size="sm", min_width = 1)
            add_random_button = gr.Button(value="+More", size="sm", min_width=1)

        #with gr.Row():
        #    assumedirectcontrol = gr.Checkbox(
        #        label="BYPASS SAFETY PROTOCOLS", value=False
        #    )
        #    add_ctrl("obp_assume_direct_control", assumedirectcontrol)
        
        # Part of presets
        with gr.Row():
                OBP_preset = gr.Dropdown(
                    label="One Button Preset",
                    choices=[OBPresets.RANDOM_PRESET_OBP] + list(OBPresets.opb_presets.keys())
                    + [OBPresets.CUSTOM_OBP],
                    value=settings["OBP_preset"],
                )
                add_ctrl("OBP_preset", OBP_preset)
        

                
        with gr.Group(visible=False) as maingroup:
            with gr.Row():
                    obp_preset_name = gr.Textbox(
                        show_label=False,
                        placeholder="Name of new preset",
                        interactive=True,
                        visible=True,
                    )
                    obp_preset_save = gr.Button(
                        value="Save as preset",
                        visible=True,
                    )
        
        # End of this part of presets
        
            with gr.Row():
                insanitylevel = gr.Slider(
                    1,
                    10,
                    value=custom_obp_values["insanitylevel"],
                    step=1,
                    label="Higher levels increases complexity and randomness of generated prompt",
                )
                add_ctrl("obp_insanitylevel", insanitylevel)
            with gr.Row():
                with gr.Column(scale=1, variant="compact"):
                    subject = gr.Dropdown(subjects, label="Subject Types", value=custom_obp_values["subject"])
                    add_ctrl("obp_subject", subject)
                with gr.Column(scale=1, variant="compact"):
                    artist = gr.Dropdown(artists, label="Artists", value=custom_obp_values["artist"])
                    add_ctrl("obp_artist", artist)

            with gr.Row():
                chosensubjectsubtypeobject = gr.Dropdown(
                    subjectsubtypesobject,
                    label="Type of object",
                    value=custom_obp_values["chosensubjectsubtypeobject"],
                    visible=False,
                )
                add_ctrl("obp_chosensubjectsubtypeobject", chosensubjectsubtypeobject)
                chosensubjectsubtypehumanoid = gr.Dropdown(
                    subjectsubtypeshumanoid,
                    label="Type of humanoids",
                    value=custom_obp_values["chosensubjectsubtypehumanoid"],
                    visible=False,
                )
                add_ctrl("obp_chosensubjectsubtypehumanoid", chosensubjectsubtypehumanoid)
                chosensubjectsubtypeconcept = gr.Dropdown(
                    subjectsubtypesconcept,
                    label="Type of concept",
                    value=custom_obp_values["chosensubjectsubtypeconcept"],
                    visible=False,
                )
                add_ctrl("obp_chosensubjectsubtypeconcept", chosensubjectsubtypeconcept)
                chosengender = gr.Dropdown(
                    genders, label="gender", value=custom_obp_values["chosengender"], visible=False
                )
                add_ctrl("obp_chosengender", chosengender)
            with gr.Row():
                with gr.Column(scale=2, variant="compact"):
                    imagetype = gr.Dropdown(imagetypes, label="type of image", value=custom_obp_values["imagetype"])
                    add_ctrl("obp_imagetype", imagetype)
                with gr.Column(scale=2, variant="compact"):
                    imagemodechance = gr.Slider(
                        1,
                        100,
                        value=custom_obp_values["imagemodechance"],
                        step=1,
                        label="One in X chance to use special image type mode",
                    )
                    add_ctrl("obp_imagemodechance", imagemodechance)
            with gr.Row():
                gr.Markdown(
                    """
                            <font size="2">
                            Override options (choose the related subject type first for better results)
                            </font>
                            """
                )
            with gr.Row():
                givensubject = gr.Textbox(label="Overwrite subject: ", value=custom_obp_values["givensubject"])
                add_ctrl("obp_givensubject", givensubject)
                smartsubject = gr.Checkbox(label="Smart subject", value=custom_obp_values["smartsubject"])
                add_ctrl("obp_smartsubject", smartsubject)
                givenoutfit = gr.Textbox(label="Overwrite outfit: ", value=custom_obp_values["givenoutfit"])
                add_ctrl("obp_givenoutfit", givenoutfit)
            with gr.Row():
                gr.Markdown(
                    """
                            <font size="2">
                            Prompt fields
                            </font>
                            """
                )
            with gr.Row():
                with gr.Column():
                    prefixprompt = gr.Textbox(
                        label="Place this in front of generated prompt (prefix)", value=custom_obp_values["prefixprompt"]
                    )
                    add_ctrl("obp_prefixprompt", prefixprompt)
                    suffixprompt = gr.Textbox(
                        label="Place this at back of generated prompt (suffix)", value=custom_obp_values["suffixprompt"]
                    )
                    add_ctrl("obp_suffixprompt", suffixprompt)
            with gr.Row():
                gr.Markdown(
                    """
                            <font size="2">
                            Additional options
                            </font>
                            """
                )
            with gr.Row():
                giventypeofimage = gr.Textbox(label="Overwrite type of image: ", value=custom_obp_values["giventypeofimage"])
                add_ctrl("obp_giventypeofimage", giventypeofimage)
            with gr.Row():
                with gr.Column():
                    antistring = gr.Textbox(
                        label="Filter out following properties (comma seperated). Example "
                        "film grain, purple, cat"
                        " ", value=custom_obp_values["antistring"]
                    )
                    add_ctrl("obp_antistring", antistring)
        with gr.Row():
            promptenhance = gr.Dropdown(
                choices=promptenhancelist, label="HYPERPROMPTING", value="none"
            )
            add_ctrl("OBP_promptenhance", promptenhance)
            
            modeltype = gr.Dropdown(
                choices=modeltypelist, label="Model type", value="SDXL"
            )
            add_ctrl("OBP_modeltype", modeltype)
        #with gr.Row():
        #    gr.Markdown(
        #        """
        #        Proud to be powered by [One Button Prompt](https://github.com/AIrjen/OneButtonPrompt)
        #        """
        #    )

        obp_outputs = [
                    obp_preset_name,
                    obp_preset_save,
                    insanitylevel,
                    subject,
                    artist,
                    chosensubjectsubtypeobject,
                    chosensubjectsubtypehumanoid,
                    chosensubjectsubtypeconcept,
                    chosengender,
                    imagetype,
                    imagemodechance,
                    givensubject,
                    smartsubject,
                    givenoutfit,
                    prefixprompt,
                    suffixprompt,
                    giventypeofimage,
                    antistring,
                ]

                
        def act_obp_preset_save(
                    obp_preset_name,
                    obp_preset_save,
                    insanitylevel,
                    subject,
                    artist,
                    chosensubjectsubtypeobject,
                    chosensubjectsubtypehumanoid,
                    chosensubjectsubtypeconcept,
                    chosengender,
                    imagetype,
                    imagemodechance,
                    givensubject,
                    smartsubject,
                    givenoutfit,
                    prefixprompt,
                    suffixprompt,
                    giventypeofimage,
                    antistring,
                ):
                    if obp_preset_name != "":
                        obp_options = OBPresets.load_obp_presets()
                        opts = {
                            "insanitylevel": insanitylevel,
                            "subject": subject,
                            "artist": artist,
                            "chosensubjectsubtypeobject": chosensubjectsubtypeobject,
                            "chosensubjectsubtypehumanoid": chosensubjectsubtypehumanoid,
                            "chosensubjectsubtypeconcept": chosensubjectsubtypeconcept,
                            "chosengender": chosengender,
                            "imagetype": imagetype,
                            "imagemodechance": imagemodechance,
                            "givensubject": givensubject,
                            "smartsubject": smartsubject,
                            "givenoutfit": givenoutfit,
                            "prefixprompt": prefixprompt,
                            "suffixprompt": suffixprompt,
                            "giventypeofimage": giventypeofimage,
                            "antistring": antistring
                        }
                        obp_options[obp_preset_name] = opts
                        OBPresets.save_obp_preset(obp_options)
                        choices = [OBPresets.RANDOM_PRESET_OBP] + list(obp_options.keys()) + [
                            OBPresets.CUSTOM_OBP
                        ]
                        return gr.update(choices=choices, value=obp_preset_name)
                    else:
                        return gr.update()

        obp_preset_save.click(act_obp_preset_save,
                    inputs=obp_outputs,
                    outputs=[OBP_preset],
                )
        
        
        
        
        def obppreset_changed(selection):
                if selection == OBPresets.CUSTOM_OBP:
                    return (
                        [obp_preset_name.update(value="", visible=True)]
                        + [maingroup.update(visible=True)]
                    )
    
                else:
                    return (
                        [obp_preset_name.update(visible=False)]
                        + [maingroup.update(visible=False)]
                    )
        OBP_preset.change(obppreset_changed,
                inputs=[OBP_preset],
                outputs=[obp_preset_name] + [maingroup]
            )
        
        
        
        
        def OBPPreset_changed_update_custom(selection):
                # Skip if Custom was selected
                if selection == OBPresets.CUSTOM_OBP:
                    return [gr.update()] * 16

                # Update Custom values based on selected One Button preset
                if selection == OBPresets.RANDOM_PRESET_OBP:
                    selected_opb_preset = OBPresets.get_obp_preset("Standard")
                else:     
                    selected_opb_preset = OBPresets.get_obp_preset(selection)
                return [
                    insanitylevel.update(value=selected_opb_preset["insanitylevel"]),
                    subject.update(value=selected_opb_preset["subject"]),
                    artist.update(value=selected_opb_preset["artist"]),
                    chosensubjectsubtypeobject.update(value=selected_opb_preset["chosensubjectsubtypeobject"]),
                    chosensubjectsubtypehumanoid.update(value=selected_opb_preset["chosensubjectsubtypehumanoid"]),
                    chosensubjectsubtypeconcept.update(value=selected_opb_preset["chosensubjectsubtypeconcept"]),
                    chosengender.update(value=selected_opb_preset["chosengender"]),
                    imagetype.update(value=selected_opb_preset["imagetype"]),
                    imagemodechance.update(value=selected_opb_preset["imagemodechance"]),
                    givensubject.update(value=selected_opb_preset["givensubject"]),
                    smartsubject.update(value=selected_opb_preset["smartsubject"]),
                    givenoutfit.update(value=selected_opb_preset["givenoutfit"]),
                    prefixprompt.update(value=selected_opb_preset["prefixprompt"]),
                    suffixprompt.update(value=selected_opb_preset["suffixprompt"]),
                    giventypeofimage.update(value=selected_opb_preset["giventypeofimage"]),
                    antistring.update(value=selected_opb_preset["antistring"]),
                ]
        OBP_preset.change(OBPPreset_changed_update_custom,
                inputs=[OBP_preset],
                outputs=[insanitylevel] + 
                [subject] + 
                [artist] + 
                [chosensubjectsubtypeobject] + 
                [chosensubjectsubtypehumanoid] + 
                [chosensubjectsubtypeconcept] + 
                [chosengender] + 
                [imagetype] + 
                [imagemodechance] + 
                [givensubject] + 
                [smartsubject] + 
                [givenoutfit] +
                [prefixprompt] +
                [suffixprompt] +
                [giventypeofimage] +
                [antistring], 
        )
        
        
        # turn things on and off for gender
        def subjectsvalue(subject):
            enable = "human" in subject
            return {
                chosengender: gr.update(visible=enable),
            }

        subject.change(subjectsvalue, [subject], [chosengender])

        # turn things on and off for subject subtype object
        def subjectsvalueforsubtypeobject(subject):
            enable = subject == "object"
            return {
                chosensubjectsubtypeobject: gr.update(visible=enable),
            }

        subject.change(
            subjectsvalueforsubtypeobject, [subject], [chosensubjectsubtypeobject]
        )

        # turn things on and off for subject subtype humanoid
        def subjectsvalueforsubtypeobject(subject):
            enable = subject == "humanoid"
            return {
                chosensubjectsubtypehumanoid: gr.update(visible=enable),
            }

        subject.change(
            subjectsvalueforsubtypeobject, [subject], [chosensubjectsubtypehumanoid]
        )

        # turn things on and off for subject subtype concept
        def subjectsvalueforsubtypeconcept(subject):
            enable = subject == "concept"
            return {
                chosensubjectsubtypeconcept: gr.update(visible=enable),
            }

        subject.change(
            subjectsvalueforsubtypeconcept, [subject], [chosensubjectsubtypeconcept]
        )

        # turn things on and off for ASSUME DIRECT CONTROL
        def assumedirectcontrolflip(assumedirectcontrol):
            enable = not assumedirectcontrol
            return {
                instant_obp: gr.update(visible=enable),
                random_button: gr.update(visible=enable),
                add_random_button: gr.update(visible=enable),
            }

        #assumedirectcontrol.change(
        #    assumedirectcontrolflip,
        #    [assumedirectcontrol],
        #    [instant_obp, random_button, add_random_button],
        #)

        #instant_obp.click(
        #    instant_gen_prompt,
        #    inputs=[
        #        insanitylevel,
        #        subject,
        #        artist,
        #        imagetype,
        #        antistring,
        #        prefixprompt,
        #        suffixprompt,
        #        givensubject,
        #        smartsubject,
        #        giventypeofimage,
        #        imagemodechance,
        #        chosengender,
        #        chosensubjectsubtypeobject,
        #        chosensubjectsubtypehumanoid,
        #        chosensubjectsubtypeconcept,
        #        givenoutfit,
        #        OBP_preset,
        #        promptenhance,
        #        modeltype,
        #        run_event,
        #    ],
        #    outputs=[prompt, run_event],
        #)
        random_button.click(
            gen_prompt,
            inputs=[
                insanitylevel,
                subject,
                artist,
                imagetype,
                antistring,
                prefixprompt,
                suffixprompt,
                givensubject,
                smartsubject,
                giventypeofimage,
                imagemodechance,
                chosengender,
                chosensubjectsubtypeobject,
                chosensubjectsubtypehumanoid,
                chosensubjectsubtypeconcept,
                givenoutfit,
                OBP_preset,
                promptenhance,
                modeltype,
            ],
            outputs=[prompt],
        )
        add_random_button.click(
            add_prompt,
            inputs=[
                prompt,
                insanitylevel,
                subject,
                artist,
                imagetype,
                antistring,
                prefixprompt,
                suffixprompt,
                givensubject,
                smartsubject,
                giventypeofimage,
                imagemodechance,
                chosengender,
                chosensubjectsubtypeobject,
                chosensubjectsubtypehumanoid,
                chosensubjectsubtypeconcept,
                givenoutfit,
                OBP_preset,
                promptenhance,
                modeltype,
            ],
            outputs=[prompt],
        )

