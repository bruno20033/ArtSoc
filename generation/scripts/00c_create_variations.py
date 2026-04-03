import pandas as pd

INPUT_FILE = "generation/data/gss2024_personas.csv"


def split_into_sentences(text):
    if pd.isna(text):
        return []
    parts = [part.strip() for part in str(text).split(".")]
    return [part + "." for part in parts if part]


def is_demographic(sentence):
    demographic_starts = [
        "I am ",
        "I have never been married.",
        "My family income is ",
        "I completed ",
        "I had ",
    ]

    # Exclude political and religious sentences that also start with "I am"
    political_prefixes = [
        "I am a strong Democrat.",
        "I am a not very strong Democrat.",
        "I am a strong Republican.",
        "I am a not very strong Republican.",
        "I am an independent",
        "I am extremely liberal.",
        "I am liberal.",
        "I am slightly liberal.",
        "I am moderate",
        "I am conservative.",
        "I am slightly conservative.",
        "I am extremely conservative.",
    ]

    religious_prefixes = [
        "I attend religious services",
        "I never attend religious services.",
    ]

    for prefix in political_prefixes + religious_prefixes:
        if sentence.startswith(prefix):
            return False

    # Demographic patterns
    if "years old" in sentence:
        return True

    if sentence in [
        "I am male.",
        "I am female.",
        "I am White.",
        "I am Black or African American.",
        "I am of another race.",
        "I am married.",
        "I am divorced.",
        "I am widowed.",
        "I am separated.",
    ]:
        return True

    if sentence.startswith("I have never been married."):
        return True

    if sentence.startswith("My family income is "):
        return True

    if sentence.startswith("I completed ") or sentence.startswith("I had "):
        return True

    return False


def is_political(sentence):
    political_prefixes = [
        "I am a strong Democrat.",
        "I am a not very strong Democrat.",
        "I am a strong Republican.",
        "I am a not very strong Republican.",
        "I am an independent",
        "I am extremely liberal.",
        "I am liberal.",
        "I am slightly liberal.",
        "I am moderate",
        "I am conservative.",
        "I am slightly conservative.",
        "I am extremely conservative.",
    ]
    return any(sentence.startswith(prefix) for prefix in political_prefixes)


def is_religious(sentence):
    return (
        sentence.startswith("I attend religious services")
        or sentence.startswith("I never attend religious services.")
    )


def build_persona(sentences, keep_function):
    kept = [s for s in sentences if keep_function(s)]
    return " ".join(kept)


def main():
    df = pd.read_csv(INPUT_FILE)

    # Variation 1: only demographics
    df_demographics = df.copy()
    df_demographics["persona"] = df_demographics["persona"].apply(
        lambda x: build_persona(split_into_sentences(x), is_demographic)
    )
    df_demographics.to_csv("personas_demographics.csv", index=False)

    # Variation 2: demographics + political opinion
    df_demo_political = df.copy()
    df_demo_political["persona"] = df_demo_political["persona"].apply(
        lambda x: " ".join(
            [
                s
                for s in split_into_sentences(x)
                if is_demographic(s) or is_political(s)
            ]
        )
    )
    df_demo_political.to_csv("personas_demographics_political.csv", index=False)

    # Variation 3: only religiousness
    df_religious = df.copy()
    df_religious["persona"] = df_religious["persona"].apply(
        lambda x: build_persona(split_into_sentences(x), is_religious)
    )
    df_religious.to_csv("personas_religiousness.csv", index=False)

    print("Created:")
    print("- personas_demographics.csv")
    print("- personas_demographics_political.csv")
    print("- personas_religiousness.csv")


if __name__ == "__main__":
    main()