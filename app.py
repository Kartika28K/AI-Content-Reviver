import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load .env file (for local development)
load_dotenv()


# ----------------------------------------------------
# Page config
# ----------------------------------------------------

st.set_page_config(
    page_title="AI Content Reviver",
    page_icon="✨",
    layout="centered",
)


# ----------------------------------------------------
# Groq Client
# ----------------------------------------------------

@st.cache_resource
def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error(
            "GROQ_API_KEY is not set. Add it to your .env file.",
            icon="🔑",
        )
        st.stop()

    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

PLATFORM_RULES = {
    "LinkedIn": {
        "description": "LinkedIn — a professional network read by peers, recruiters, and industry leaders.",
        "style": (
            "Write in short paragraphs separated by blank lines for easy scanning. "
            "Open with a strong first line that stands alone as a hook. "
            "Use 1–3 relevant hashtags at the very end only. "
            "No bullet-point lists unless they add real clarity."
        ),
    },
    "Instagram": {
        "description": "Instagram — a visual, lifestyle-driven platform where captions complement an image.",
        "style": (
            "Start with an eye-catching first line (readers see only the first 1–2 lines before 'more'). "
            "Use line breaks generously to create visual breathing room. "
            "End with a blank line then 6–10 tightly relevant hashtags on their own line."
        ),
    },
    "Twitter": {
        "description": "Twitter / X — a micro-blogging platform where brevity wins.",
        "style": (
            "HARD LIMIT: 280 characters total — count every character including spaces and hashtags. "
            "Trim ruthlessly until it fits. "
            "Hook the reader in the very first 5 words. "
            "Use at most 1–2 hashtags. "
            "Output exactly one tweet — no thread, no commentary."
        ),
    },
}

TONE_PROFILES = {
    "Professional": {
        "system_role": (
            "You are a senior corporate communications strategist who writes polished, "
            "authoritative content for C-suite executives and industry publications."
        ),
        "rules": (
            "- Use formal, precise language. No contractions (write 'do not', not 'don't').\n"
            "- Structure ideas logically: context → insight → implication.\n"
            "- Use industry-appropriate vocabulary; sound like a subject-matter expert.\n"
            "- Sentences are complete and grammatically perfect.\n"
            "- Avoid exclamation marks — let the ideas speak for themselves.\n"
            "- Example feel: a Harvard Business Review op-ed condensed for social."
        ),
        "temperature": 0.5,
    },
    "Friendly": {
        "system_role": (
            "You are a warm, personable brand voice specialist who writes content that "
            "feels like advice from a knowledgeable friend — approachable, genuine, and encouraging."
        ),
        "rules": (
            "- Use contractions freely (you're, we've, it's).\n"
            "- Keep the original narrator's voice; do NOT switch from first-person to second-person.\n"
            "- Warm openers: start with something relatable or empathetic, still in the author's voice.\n"
            "- Short, conversational sentences mixed with slightly longer ones.\n"
            "- End with an inclusive, encouraging note that reflects the author's experience.\n"
            "- Example feel: a trusted colleague sharing their own story in a warm, open way."
        ),
        "temperature": 0.7,
    },
    "Casual": {
        "system_role": (
            "You are a laid-back social media creator who talks to followers like they're "
            "close friends — real, relaxed, and completely free of corporate polish."
        ),
        "rules": (
            "- Write exactly how people talk: contractions, informal phrases, even sentence fragments.\n"
            "- Short punchy sentences. Get to the point fast.\n"
            "- Slang and colloquialisms are fine (ngl, lowkey, literally, etc.).\n"
            "- Avoid anything that sounds like a press release or corporate memo.\n"
            "- It's OK to be a little playful or cheeky.\n"
            "- Example feel: a friend texting you something cool they just learned."
        ),
        "temperature": 0.85,
    },
    "Viral": {
        "system_role": (
            "You are a viral content strategist who has written posts with millions of impressions. "
            "You know exactly how to stop the scroll, trigger an emotional response, and make people "
            "share without thinking twice."
        ),
        "rules": (
            "- Line 1 MUST be a scroll-stopping hook: a bold claim, shocking stat, hot take, or open loop.\n"
            "- Use pattern interrupts: start with an unexpected angle, not the obvious one.\n"
            "- Short sentences. Punchy. Sometimes just one word per line for impact.\n"
            "- Inject ONE strong emotion: surprise, curiosity, inspiration, or mild controversy.\n"
            "- Build tension or curiosity that only the final line resolves.\n"
            "- End with a clear, urgent call-to-action (share, comment, save, tag someone).\n"
            "- Use power words: 'never', 'always', 'most people', 'nobody talks about', 'here's the truth'.\n"
            "- Example feel: a tweet that makes someone stop mid-scroll and think 'I have to share this'."
        ),
        "temperature": 0.95,
    },
}

LENGTH_RULES = {
    "Short": (
        "LENGTH: Write exactly 1 short paragraph. Be concise and punchy — every word must earn its place. "
        "No multi-paragraph structure."
    ),
    "Medium": (
        "LENGTH: Write 2–3 paragraphs. Give the idea room to breathe without overstaying its welcome. "
        "Each paragraph should add something new."
    ),
    "Long": (
        "LENGTH: Write a detailed, in-depth post. Develop the idea fully with supporting points, "
        "context, and a strong close. Longer is fine — depth is the goal here."
    ),
}

EMOJI_RULES = {
    "Add Emojis ✅": (
        "EMOJIS: Naturally include relevant emojis throughout the post to add personality and "
        "visual interest. Place them where they feel organic, not forced."
    ),
    "Remove Emojis ❌": (
        "EMOJIS: Use ABSOLUTELY ZERO emojis anywhere in the post. "
        "Not a single one — not in the body, not at the end, nowhere. This is non-negotiable."
    ),
}

POV_RULES = """
PERSPECTIVE RULES — these override everything else and must never be broken:
1. Preserve the original point of view exactly. If the input uses "I", "my", or "me", the output MUST also use "I", "my", "me".
2. Never convert first-person ("I did X") into second-person ("You did X" / "You've taken a step…").
3. Never reframe the post as advice, encouragement, coaching, or a message addressed to the reader.
4. Never invent achievements, facts, numbers, or details not present in the original.
5. Do not add questions unless the original post contains one.
6. Rewrite only — improve style, clarity, and platform fit. Do not change who is speaking or what happened.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt(tone: str) -> str:
    profile = TONE_PROFILES[tone]
    return (
        f"{profile['system_role']}\n\n"
        f"{POV_RULES}\n"
        "Output ONLY the finished post — no preamble, no explanation, no meta-commentary. "
        "Never start your reply with phrases like 'Here is' or 'Sure!' — go straight to the post."
    )


def build_user_prompt(content: str, platform: str, tone: str, length: str, emoji_mode: str) -> str:
    plat = PLATFORM_RULES[platform]
    tone_profile = TONE_PROFILES[tone]
    # Twitter is always a single 280-char tweet — length selector doesn't apply
    effective_length = "Short" if platform == "Twitter" else length
    length_rule = LENGTH_RULES[effective_length]
    if platform == "Twitter" and length != "Short":
        length_rule += (
            " (Note: Twitter's 280-character hard limit overrides the length selector — "
            "Short is enforced automatically.)"
        )
    emoji_rule = EMOJI_RULES[emoji_mode]

    return f"""Rewrite the content below for {platform} in a {tone} tone.

=== PLATFORM: {plat['description']} ===
Platform style rules:
{plat['style']}

=== TONE: {tone.upper()} ===
Tone rules — follow every one of these strictly:
{tone_profile['rules']}

=== LENGTH (NON-NEGOTIABLE) ===
{length_rule}

=== EMOJI RULE (NON-NEGOTIABLE — overrides all other emoji guidance) ===
{emoji_rule}

=== PERSPECTIVE (NON-NEGOTIABLE) ===
{POV_RULES}

=== ORIGINAL CONTENT ===
{content}

=== YOUR TASK ===
Rewrite the content above so it sounds unmistakably {tone} and is perfectly formatted for {platform}.
The platform rules, tone rules, length rule, and emoji rule must ALL be reflected in the output.
Keep the original narrator's voice — if it is written in first person, keep it in first person.
Do NOT blend tones — commit fully to {tone}. Output the post now:"""


def build_hooks_prompt(rewritten: str, platform: str, tone: str) -> str:
    return f"""Here is a rewritten {platform} post in a {tone} tone:

---
{rewritten}
---

Generate exactly 5 engaging opening hooks for this post.

Rules:
- Each hook replaces only the opening line of the post — keep them SHORT (one sentence max).
- Match the {platform} platform style and {tone} tone exactly.
- Do NOT change the meaning or invent new facts.
- Each hook must spark curiosity and make the reader want to keep reading.
- Number them 1–5. No extra commentary — just the 5 hooks.
- Do not include the full post, only the hook lines."""


def build_suggestions_prompt(rewritten: str, platform: str, tone: str) -> str:
    return f"""You are a sharp social media editor. Analyze this {platform} post written in a {tone} tone:

---
{rewritten}
---

Give exactly 4 actionable suggestions to improve this post.

Focus areas (one suggestion each, in this exact order):
1. Opening Hook — how to make the first line more compelling
2. Engagement — how to drive more comments, shares, or saves
3. CTA (Call to Action) — how to make the closing action clearer or stronger
4. Clarity — one thing that could be said more clearly or concisely

Format:
1. [Opening Hook] Your suggestion here.
2. [Engagement] Your suggestion here.
3. [CTA] Your suggestion here.
4. [Clarity] Your suggestion here.

Rules:
- Each suggestion must be ONE sentence. No exceptions.
- Be specific — reference the actual content, not generic advice.
- No preamble. Start directly with "1."."""


def build_hashtags_prompt(rewritten: str, platform: str, tone: str) -> str:
    return f"""You are a hashtag strategist. Here is a {platform} post in a {tone} tone:

---
{rewritten}
---

Generate 4 groups of hashtags based on this post. Each group contains exactly 5 unique hashtags.

Groups:
**Professional** — industry-standard, credibility-building hashtags used by professionals.
**Trending** — currently popular hashtags that are riding a wave of high engagement.
**Niche** — specific, targeted hashtags for a smaller but highly engaged community.
**Viral** — broad, high-volume hashtags that maximize reach and shareability.

Rules:
- No duplicates across ANY group — every single hashtag must be unique across all 4 groups.
- Include the # symbol on every hashtag.
- Output format exactly as shown below. No other commentary.

**Professional**
#tag1 #tag2 #tag3 #tag4 #tag5

**Trending**
#tag1 #tag2 #tag3 #tag4 #tag5

**Niche**
#tag1 #tag2 #tag3 #tag4 #tag5

**Viral**
#tag1 #tag2 #tag3 #tag4 #tag5"""


# ---------------------------------------------------------------------------
# Groq callers
# ---------------------------------------------------------------------------

def call_groq(client: Groq, system: str, user: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("✨ AI Content Reviver")
st.caption("Paste old social media content → choose your platform & tone → get an AI-polished rewrite instantly.")

st.divider()

# --- Inputs -----------------------------------------------------------------
original = st.text_area(
    "📋 Paste your original content",
    placeholder="Paste a LinkedIn post, tweet, caption, or any social media copy here…",
    height=200,
    key="original_input",
)

col1, col2 = st.columns(2)
with col1:
    platform = st.selectbox(
        "🌐 Target platform",
        ["LinkedIn", "Instagram", "Twitter"],
        key="platform_select",
    )
with col2:
    tone = st.selectbox(
        "🎭 Tone",
        ["Professional", "Friendly", "Casual", "Viral"],
        key="tone_select",
    )

col3, col4 = st.columns(2)
with col3:
    length = st.radio(
        "📏 Length",
        ["Short", "Medium", "Long"],
        horizontal=True,
        key="length_radio",
    )
with col4:
    emoji_mode = st.radio(
        "😀 Emojis",
        ["Add Emojis ✅", "Remove Emojis ❌"],
        horizontal=True,
        key="emoji_radio",
    )

revive_clicked = st.button("✨ Revive Content", type="primary", use_container_width=True)

# --- Generation -------------------------------------------------------------
if revive_clicked:
    if not original.strip():
        st.warning("Please paste some content before clicking Revive.", icon="⚠️")
    else:
        client = get_client()

        # Clear all stale results before generating new ones
        for key in ("rewritten", "hooks", "suggestions", "hashtags",
                    "last_platform", "last_tone", "last_length", "last_emoji"):
            st.session_state.pop(key, None)

        try:
            with st.spinner("✍️ Rewriting your content…"):
                rewritten = call_groq(
                    client,
                    system=build_system_prompt(tone),
                    user=build_user_prompt(original, platform, tone, length, emoji_mode),
                    temperature=TONE_PROFILES[tone]["temperature"],
                    max_tokens=1024,
                )
                st.session_state["rewritten"] = rewritten
                # Save context labels immediately so they're accurate even if aux calls fail
                st.session_state["last_platform"] = platform
                st.session_state["last_tone"] = tone
                st.session_state["last_length"] = length
                st.session_state["last_emoji"] = emoji_mode

            with st.spinner("🎣 Generating opening hooks…"):
                hooks = call_groq(
                    client,
                    system=(
                        "You are a scroll-stopping headline writer. "
                        "You generate short, punchy opening hooks that match a specific platform and tone exactly. "
                        "Output only the numbered list — no preamble, no commentary."
                    ),
                    user=build_hooks_prompt(rewritten, platform, tone),
                    temperature=0.9,
                    max_tokens=512,
                )
                st.session_state["hooks"] = hooks

            with st.spinner("💡 Analyzing content for suggestions…"):
                suggestions = call_groq(
                    client,
                    system=(
                        "You are a sharp, direct social media editor. "
                        "You give concise, actionable feedback — never vague or generic. "
                        "Output only the 4 numbered suggestions. No preamble."
                    ),
                    user=build_suggestions_prompt(rewritten, platform, tone),
                    temperature=0.5,
                    max_tokens=512,
                )
                st.session_state["suggestions"] = suggestions

            with st.spinner("#️⃣ Building hashtag groups…"):
                hashtags = call_groq(
                    client,
                    system=(
                        "You are a hashtag strategist who creates targeted, non-duplicate hashtag sets. "
                        "Output only the formatted groups — no preamble, no explanation."
                    ),
                    user=build_hashtags_prompt(rewritten, platform, tone),
                    temperature=0.6,
                    max_tokens=512,
                )
                st.session_state["hashtags"] = hashtags

        except Exception as exc:
            st.error(f"Groq API error: {exc}", icon="❌")

# --- Output (persists via session state) ------------------------------------
if "rewritten" in st.session_state and st.session_state["rewritten"]:
    rewritten = st.session_state["rewritten"]
    last_platform = st.session_state.get("last_platform", platform)
    last_tone = st.session_state.get("last_tone", tone)
    last_length = st.session_state.get("last_length", "Medium")
    last_emoji = st.session_state.get("last_emoji", "Add Emojis ✅")

    st.divider()
    st.subheader(f"🎯 Rewritten for {last_platform} — {last_tone} · {last_length} · {last_emoji}")

    # Main rewrite
    st.code(rewritten, language=None)
    st.caption("☝️ Click the copy icon in the top-right corner of the code block above to copy.")

    # Extra results in tabs
    hooks_text = st.session_state.get("hooks", "")
    suggestions_text = st.session_state.get("suggestions", "")
    hashtags_text = st.session_state.get("hashtags", "")

    if hooks_text or suggestions_text or hashtags_text:
        st.divider()
        tab_hooks, tab_suggestions, tab_hashtags = st.tabs(
            ["🎣 Opening Hooks", "💡 Suggestions", "#️⃣ Hashtags"]
        )

        with tab_hooks:
            st.markdown("**5 alternative opening hooks** — drop any one of these in as your first line:")
            for line in hooks_text.splitlines():
                line = line.strip()
                if line:
                    st.markdown(line)

        with tab_suggestions:
            st.markdown("**4 actionable suggestions** to sharpen this post:")
            for line in suggestions_text.splitlines():
                line = line.strip()
                if line:
                    st.markdown(line)

        with tab_hashtags:
            st.markdown("**4 hashtag groups** — pick the set that fits your goal:")
            # Use code block so '#tag' is never interpreted as a Markdown heading
            st.code(hashtags_text, language=None)

    st.divider()
    if st.button("🔄 Revive Again", use_container_width=True):
        for key in ("rewritten", "hooks", "suggestions", "hashtags",
                    "last_platform", "last_tone", "last_length", "last_emoji"):
            st.session_state.pop(key, None)
        st.rerun()
