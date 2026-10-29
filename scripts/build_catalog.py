"""Script to build and validate the central datasets/sign_language_catalog.json."""

import json
from pathlib import Path


def build_catalog():
    catalog_path = Path("datasets/sign_language_catalog.json")
    entries = []

    # 1. ASL Alphabet (26 letters, A-Z)
    alpha_descriptions = {
        "A": (
            "Fist with thumb resting along the side of the index finger, palm facing forward.",
            "Keep the thumb upright against index finger side; do not tuck inside palm.",
        ),
        "B": (
            "Flat hand with four fingers straight up and pressed together, thumb folded across palm.",
            "Fingers remain vertical and parallel, palm facing outward.",
        ),
        "C": (
            "Curved hand forming a 'C' shape with fingers and thumb, palm facing sideways.",
            "Ensure fingers curve smoothly like gripping a cup.",
        ),
        "D": (
            "Index finger straight up, other three fingers touching thumb tip to form an 'O'.",
            "Only the index finger is extended vertically.",
        ),
        "E": (
            "All fingers bent down with fingertips resting along the top edge of the folded thumb.",
            "Fingertips curl tightly against the thumb top.",
        ),
        "F": (
            "Index finger and thumb touching tips to form a circle; middle, ring, pinky upright.",
            "Resembles an 'OK' sign; keep the other three fingers spread.",
        ),
        "G": (
            "Index finger and thumb extended parallel pointing horizontally sideways.",
            "Fingers point across the chest in horizontal parallel lines.",
        ),
        "H": (
            "Index and middle fingers extended together horizontally pointing sideways.",
            "Both fingers stay tightly closed together pointing across body.",
        ),
        "I": (
            "Pinky finger extended upright, other fingers folded into a fist with thumb across.",
            "Only the little finger is vertical; palm faces forward.",
        ),
        "J": (
            "Pinky extended (as in I), swooping downward and curling back up in a 'J' curve in the air.",
            "Dynamic motion sign: trace letter J with the pinky tip.",
        ),
        "K": (
            "Index finger pointing up, middle finger angled forward, thumb resting at middle joint.",
            "Index and middle spread in a V; thumb rests between them.",
        ),
        "L": (
            "Index finger pointing straight up and thumb extended outward at 90 degrees.",
            "Clear L shape formed between thumb and index finger.",
        ),
        "M": (
            "Three fingers folded over the thumb, with thumb tip peeking between ring and pinky.",
            "Three fingers drape over the tucked thumb.",
        ),
        "N": (
            "Two fingers (index and middle) folded over thumb, thumb peeking between middle and ring.",
            "Two fingers drape over the tucked thumb.",
        ),
        "O": (
            "All fingers and thumb curved with tips meeting to form an 'O' circle.",
            "Smooth rounded circular shape with palm facing forward.",
        ),
        "P": (
            "K handshape oriented downward, middle finger pointing toward ground.",
            "Tilt the K sign 90 degrees down from the wrist.",
        ),
        "Q": (
            "G handshape oriented downward, index finger and thumb pointing toward ground.",
            "Tilt the G sign downward pointing at the floor.",
        ),
        "R": (
            "Index and middle fingers crossed, middle finger behind index, palm facing forward.",
            "Cross fingers tightly like wishing for good fortune.",
        ),
        "S": (
            "Fist with thumb wrapped across the front of the curled fingers.",
            "Distinct from A: thumb lies horizontally OVER front fingers.",
        ),
        "T": (
            "Fist with thumb inserted between the index and middle fingers.",
            "Thumb peeks up between index and middle knuckles.",
        ),
        "U": (
            "Index and middle fingers extended straight up and pressed together, thumb across palm.",
            "Keep both extended fingers touching; palm faces forward.",
        ),
        "V": (
            "Index and middle fingers extended straight up and spread apart in a 'V' shape.",
            "Victory / Peace sign handshape.",
        ),
        "W": (
            "Index, middle, and ring fingers extended straight up and spread apart in a 'W'.",
            "Three middle fingers spread vertically.",
        ),
        "X": (
            "Fist with index finger curled into a hooked shape.",
            "Hooked index finger like a pirate hook; palm sideways.",
        ),
        "Y": (
            "Thumb and pinky extended outward, middle three fingers folded into palm.",
            "Iconic shaka / hang-loose handshape.",
        ),
        "Z": (
            "Index finger extended, drawing a 'Z' path through the air.",
            "Dynamic motion sign: trace Z in the air with index fingertip.",
        ),
    }

    for letter, (desc, notes) in alpha_descriptions.items():
        vpath = f"datasets/samples/alphabet/{letter.lower()}.mp4"
        exists = Path(vpath).exists()
        entries.append(
            {
                "id": f"asl_alpha_{letter.lower()}",
                "name": letter,
                "language": "ASL",
                "category": "Alphabet",
                "type": "letter",
                "description": desc,
                "video": vpath if exists else None,
                "available": exists,
                "source": "SignBridge External ASL Dataset",
                "license": "CC-BY-4.0 / Educational",
                "notes": notes,
            }
        )

    # 2. Numbers (0-9)
    num_descriptions = {
        "0": ("All fingertips touching thumb tip forming a closed oval/zero shape.", "Similar to letter O; palm faces forward."),
        "1": ("Index finger straight up, other fingers curled into fist with thumb across.", "Standard number 1 counting sign in ASL."),
        "2": ("Index and middle fingers straight up and spread into a V.", "Similar to letter V; palm faces forward."),
        "3": ("Thumb, index, and middle fingers extended upright and spread; ring and pinky folded.", "Note: ASL 3 uses thumb, not ring finger."),
        "4": ("Four fingers extended upright and spread; thumb folded across palm.", "Palm faces forward with four vertical fingers."),
        "5": ("All five fingers extended upright and spread apart, palm facing forward.", "Open 5 handshape with all digits spread."),
        "6": ("Thumb and pinky finger touching tips; index, middle, and ring fingers extended upright.", "In ASL counting, 6 connects thumb and pinky."),
        "7": ("Thumb and ring finger touching tips; index, middle, and pinky fingers extended upright.", "Connect thumb and ring finger tip."),
        "8": ("Thumb and middle finger touching tips; index, ring, and pinky fingers extended upright.", "Connect thumb and middle finger tip."),
        "9": ("Thumb and index finger touching tips; middle, ring, and pinky fingers extended upright.", "Similar to letter F / OK sign."),
    }

    for num, (desc, notes) in num_descriptions.items():
        entries.append(
            {
                "id": f"asl_num_{num}",
                "name": num,
                "language": "ASL",
                "category": "Numbers",
                "type": "number",
                "description": desc,
                "video": None,
                "available": False,
                "source": "ASL Reference Library",
                "license": "Educational Reference",
                "notes": notes,
            }
        )

    # 3. Greetings
    greetings = [
        (
            "Hello",
            "Open flat B hand at temple, moving outward and slightly upward in a salute gesture.",
            "datasets/samples/greetings/hello.mp4",
            "Universal ASL greeting, 1-hand dominant gesture.",
            "asl_greet_hello",
            "SignBridge Verified Demonstration",
            "CC-BY / C-UDA Educational",
        ),
        (
            "Thank You",
            "Flat hand touches chin with fingertips and moves outward toward listener.",
            "datasets/samples/greetings/thank_you.mp4",
            "Expresses gratitude; always directed outward.",
            "asl_greet_thank_you",
            "SignBridge Verified Demonstration",
            "CC-BY / C-UDA Educational",
        ),
        (
            "Bye / Goodbye",
            "Open dominant hand (5-hand/flat hand) raised up in front of the shoulder at shoulder/head height, palm facing outward. Fingers wave gently back and forth in a waving motion.",
            "datasets/samples/greetings/goodbye.mp4",
            "Educational reference: AISA School (https://www.aisa.solutions/asl/signs/bye) and Lifeprint ASL University. Standard ASL Bye is signed at shoulder/head level with open palm facing outward. Verified demonstration video sourced from American Sign Language Dataset under MIT License.",
            "asl_greet_goodbye",
            "American Sign Language (ASL) Dataset (MIT License) & AISA Educational Reference",
            "MIT License / Educational Reference",
        ),
        (
            "Good Morning",
            "Compound sign: Good (chin to open palm) followed by Morning (dominant hand rises under non-dominant arm).",
            None,
            "Compound sign: GOOD + MORNING.",
            "asl_greet_good_morning",
            "ASL Reference Library",
            "CC-BY / C-UDA Educational",
        ),
        (
            "Good Night",
            "Compound sign: Good followed by Night (dominant curved hand arches over horizontal non-dominant wrist).",
            None,
            "Compound sign: GOOD + NIGHT.",
            "asl_greet_good_night",
            "ASL Reference Library",
            "CC-BY / C-UDA Educational",
        ),
        (
            "Nice to Meet You",
            "Flat dominant hand slides across non-dominant palm (Nice), then index fingers meet (Meet).",
            None,
            "Compound sign: NICE + MEET + YOU.",
            "asl_greet_nice_to_meet_you",
            "ASL Reference Library",
            "CC-BY / C-UDA Educational",
        ),
        (
            "Welcome",
            "Open flat palm facing upward, sweeping inward toward torso from the side.",
            None,
            "Gentle inviting scoop toward oneself.",
            "asl_greet_welcome",
            "ASL Reference Library",
            "CC-BY / C-UDA Educational",
        ),
        (
            "How Are You?",
            "Both hands in curved-horns shape placed against chest rotating outward (How), then pointing to listener (You).",
            None,
            "Common question: HOW + YOU with questioning facial expression.",
            "asl_greet_how_are_you",
            "ASL Reference Library",
            "CC-BY / C-UDA Educational",
        ),
    ]

    for item in greetings:
        name, desc, vpath, notes, cid, src, lic = item
        exists = vpath is not None and Path(vpath).exists()
        entries.append(
            {
                "id": cid,
                "name": name,
                "language": "ASL",
                "category": "Greetings",
                "type": "phrase" if (" " in name and " / " not in name) else "word",
                "description": desc,
                "video": vpath if exists else None,
                "available": exists,
                "source": src if not exists else "SignBridge Verified Demonstration",
                "license": lic,
                "notes": notes,
            }
        )

    # 4. Common Words
    common_words = [
        ("Hello", "Open flat B hand at temple, moving outward in a salute gesture.", "datasets/samples/common_words/hello.mp4", "Standard friendly greeting."),
        ("Help", "Closed A-fist with thumb upright rests on open flat non-dominant palm, both lifted upward together.", "datasets/samples/common_words/help.mp4", "Two-handed sign; can be directional toward recipient."),
        ("Yes", "S-fist held upright, tilting up and down at the wrist like a nodding head.", "datasets/samples/common_words/yes.mp4", "Mimics the vertical nodding motion of agreement."),
        ("No", "Index and middle fingers together snap shut against the thumb, palm facing listener.", "datasets/samples/common_words/no.mp4", "Sharp, decisive closing motion."),
        ("Stop", "Open flat dominant hand chops downward decisively onto open flat non-dominant palm.", "datasets/samples/common_words/stop.mp4", "Two-handed sign; vertical palm strikes horizontal palm."),
        ("Thank You", "Flat hand touches chin with fingertips and extends outward toward listener.", "datasets/samples/common_words/thank_you.mp4", "Expresses gratitude; always faces the other person."),
        ("Please", "Open flat dominant hand rubs in a clockwise circular motion over the chest/heart.", "datasets/samples/common_words/please.mp4", "Polite request sign; circular motion on upper torso. Verified video demonstration."),
        ("Sorry", "A-fist rubs in small circular clockwise motion over the heart/chest.", "datasets/samples/common_words/sorry.mp4", "Contrition gesture; resembles Please but made with a closed fist. Verified video demonstration."),
        ("Water", "W-handshape (three fingers up) index finger taps twice against the side of the chin or lips.", "datasets/samples/common_words/water.mp4", "Letter W taps chin twice. Verified video demonstration."),
        ("Food", "Flattened O handshape with fingertips touching, tapped twice against the mouth.", "datasets/samples/common_words/food.mp4", "Represents bringing food to the mouth; noun form. Verified video demonstration."),
        ("Eat", "Flattened O handshape brought once decisively to the mouth.", "datasets/samples/common_words/food.mp4", "Verb form of food; single deliberate motion to mouth. Shares video with Food."),
        ("Drink", "C-handshape held like a cup, tilted back toward the mouth as if drinking.", None, "Mimetic cup gesture tipped toward mouth."),
        ("More", "Both hands in flattened O handshapes, fingertips tapping together repeatedly in front of chest.", "datasets/samples/common_words/more.mp4", "Two-handed sign bringing fingertips together in center. Verified video demonstration."),
        ("Like", "Open 8 hand on chest, pulling outward while thumb and middle pinch together.", "datasets/samples/common_words/like.mp4", "Pulling a feeling out from the chest. Verified video demonstration."),
        ("Love", "Both arms crossed over chest with hands in fists against shoulders, hugging oneself.", None, "Cross arms over heart in a gentle embrace."),
        ("Want", "Both open hands palms up with curved claw fingers, pulling back toward body.", None, "Two-handed pulling motion with clawed fingers."),
        ("Need", "X-handshape bent downward repeatedly at the wrist in front of body.", None, "Downward flex of hooked index finger; also means Must."),
    ]

    for name, desc, vpath, notes in common_words:
        exists = vpath is not None and Path(vpath).exists()
        slug = name.lower().replace(" ", "_")
        cid = f"asl_word_{slug}"
        if any(e["id"] == cid for e in entries):
            cid = f"{cid}_cw"
        entries.append(
            {
                "id": cid,
                "name": name,
                "language": "ASL",
                "category": "Common Words",
                "type": "word",
                "description": desc,
                "video": vpath if exists else None,
                "available": exists,
                "source": "SignBridge Verified Demonstration" if exists else "ASL Reference Library",
                "license": "CC-BY / C-UDA Educational",
                "notes": notes,
            }
        )

    # 5. Daily Communication
    daily_comm = [
        ("Home", "Flattened O handshape touches the side of the chin, then moves to touch the upper cheek.", "Touches where you eat and where you sleep."),
        ("School", "Open flat dominant hand claps down twice horizontally onto open flat non-dominant palm.", "Two horizontal claps of flat palms."),
        ("Family", "Both hands in F-handshape starting with index/thumbs touching, curving outward in a circle until pinkies meet.", "Circle represents the family unit boundary."),
        ("Friend", "Both index fingers hooked (X-handshapes), interlocked first one way then reversed.", "Fingers link together like a chain bond."),
        ("Mother", "Open 5-handshape with thumb touching the chin, fingers spread upright.", "Lower face represents feminine signs in ASL."),
        ("Father", "Open 5-handshape with thumb touching the forehead, fingers spread upright.", "Upper face represents masculine signs in ASL."),
        ("Work", "Both hands in S-fists; dominant wrist taps twice on top of non-dominant wrist.", "Tapping fists symbolize manual labor."),
        ("Time", "Dominant index finger taps twice on the back of the non-dominant wrist where a watch is worn.", "Mimics checking a wristwatch."),
        ("Day", "Non-dominant arm held horizontally; dominant index finger upright resting on elbow, sweeping down like the sun setting.", "Arc of the sun across the sky."),
        ("Night", "Non-dominant arm horizontal; dominant curved hand drapes over the horizontal non-dominant wrist.", "The sun dropping below the horizon."),
        ("Man", "Thumb of open 5-hand touches forehead, then hand moves down to touch chest.", "Combines Father location with flat chest touch."),
        ("Woman", "Thumb of open 5-hand touches chin, then moves down to touch chest.", "Combines Mother location with flat chest touch."),
        ("Boy", "Dominant hand at forehead grabs and opens like grasping the brim of a baseball cap twice.", "Action resembles grabbing a cap visor."),
        ("Girl", "Dominant A-hand thumb strokes downward along the jawline from ear to chin.", "Mimics the bonnet string of historic dresses."),
        ("Name", "Both hands in H-handshape; dominant index/middle cross over non-dominant index/middle tapping twice.", "Fingers form an X-tap pattern."),
        ("Understand", "Dominant index finger flicks upward like a lightbulb turning on near the temple/forehead.", "Flicking finger mimics an idea popping up."),
    ]

    for name, desc, notes in daily_comm:
        slug = name.lower().replace(" ", "_")
        entries.append(
            {
                "id": f"asl_comm_{slug}",
                "name": name,
                "language": "ASL",
                "category": "Daily Communication",
                "type": "word",
                "description": desc,
                "video": None,
                "available": False,
                "source": "ASL Reference Library",
                "license": "Educational Reference",
                "notes": notes,
            }
        )

    # 6. Common Phrases
    phrases = [
        ("How are you?", "Two curved hands against chest roll outward (How), followed by index pointing forward (You).", "Facial grammar: furrow brows for Wh-questions."),
        ("I love you", "Hand with thumb, index, and pinky extended upright while middle and ring are folded (ILY handshape).", "Iconic ASL sign combining I, L, and Y letters."),
        ("See you later", "V-handshape points to eyes then flicks forward (See), points to listener (You), then L-hand tilts forward (Later).", "Casual American sign phrase."),
        ("Thank you very much", "Two-handed Thank You: both open hands touch chin and sweep forward extensively with a slight bow.", "Deep expression of appreciation."),
        ("What is your name?", "Point to person (You), sign Name (crossed H-fingers), then open 5-hands palms up shaking slightly (What).", "Word order in ASL: YOU NAME WHAT? with furrowed brows."),
        ("Need help", "Sign Need (X-hook flexes down) followed immediately by Help (fist on flat palm lifted).", "Essential emergency communication phrase."),
        ("You are welcome", "Open flat palm held in front sweeps inward toward signer in a gracious arc.", "Polite response to Thank You."),
        ("Nice to meet you", "Flat palm slides across flat base (Nice), then index fingers touch (Meet), point forward (You).", "Standard cordial greeting phrase."),
        ("Have a good day", "Compound: Good (chin to palm) followed by Day (arm arching down).", "Friendly parting wish."),
    ]

    for name, desc, notes in phrases:
        slug = name.lower().replace(" ", "_").replace("?", "")
        entries.append(
            {
                "id": f"asl_phrase_{slug}",
                "name": name,
                "language": "ASL",
                "category": "Common Phrases",
                "type": "phrase",
                "description": desc,
                "video": None,
                "available": False,
                "source": "ASL Reference Library",
                "license": "Educational Reference",
                "notes": notes,
            }
        )

    catalog = {
        "version": "1.0",
        "title": "SignBridge Comprehensive ASL Reference Library Catalog",
        "total_entries": len(entries),
        "available_videos": sum(1 for e in entries if e["available"]),
        "categories": [
            "Alphabet",
            "Numbers",
            "Greetings",
            "Common Words",
            "Daily Communication",
            "Common Phrases",
        ],
        "signs": entries,
    }

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(
        f"Catalog successfully written to {catalog_path} with {len(entries)} total entries "
        f"and {catalog['available_videos']} available video demonstrations."
    )


if __name__ == "__main__":
    build_catalog()
