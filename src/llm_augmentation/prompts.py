"""
Week 3 - Prompt 設計
兩支 prompt：user profiling / item attribute augmentation。
參考 LLMRec (Fig 2b/c) 的角色設定，但輸出格式改為結構化 JSON，方便 parser 處理。

設計重點（對應 project-plan-detailed.md 第3週步驟1-2）：
- 明確要求「只能輸出 JSON，不可有其他文字」，開源小模型特別需要這種硬性約束
- 各帶 1 個 few-shot 範例，經驗上能明顯提升格式穩定率
- 欄位刻意設計得可與既有資料對照，方便人工檢查合理性：
    - user 的 guessed_age_group 可以對照 users.dat 的真實 Age 欄位
    - item 的 supplementary_genres 可以對照 movies.dat 的既有 Genres 欄位
"""

USER_PROFILE_RATING_TENDENCY_VALUES = {"generous", "critical", "neutral"}
USER_PROFILE_AGE_GROUP_VALUES = {"under18", "18-24", "25-34", "35-44", "45+", "unknown"}

USER_PROFILE_SYSTEM = (
    "你是推薦系統的使用者偏好分析助手。"
    "根據使用者過去評分過的電影清單（片名、類型、評分），推論這位使用者的觀影偏好。"
    "你只能輸出一個JSON物件，禁止輸出任何JSON以外的文字、說明、前言或markdown code block標記（例如```）。"
    "欄位若無法從資料中判斷，preferred_genres/disliked_genres請填空陣列[]，"
    "guessed_age_group請填\"unknown\"，禁止捏造沒有根據的內容。"
    "rating_tendency欄位「只能」是以下三者之一，不可使用其他任何用詞："
    "generous（整體評分偏寬容/偏高）、critical（整體評分偏嚴格/偏低）、neutral（中庸、無明顯偏向）。"
    "請通盤考量該使用者「全部」評分紀錄的平均高低後再判斷，不要只被少數幾筆評分帶偏。"
    "請全程使用繁體中文回答，絕對不可出現任何簡體字。"
)

USER_PROFILE_FEWSHOT = """範例輸入：
使用者評分紀錄：
- Toy Story (Animation, Children's, Comedy) - 評分5
- Aladdin (Animation, Children's, Musical) - 評分5
- Lion King, The (Animation, Children's, Musical) - 評分4
- Die Hard (Action, Thriller) - 評分2

範例輸出：
{"guessed_age_group": "under18", "preferred_genres": ["Animation", "Children's", "Musical"], "disliked_genres": ["Action", "Thriller"], "rating_tendency": "generous", "profile_summary": "偏好闔家觀賞的動畫與音樂片，對動作驚悚片評價較低"}"""


def build_user_profile_prompt(user_history: list[dict]) -> str:
    """
    user_history: [{"title": str, "genres": str, "rating": int}, ...]
    依評分時間或原順序列出即可，不必額外排序（Week3只是小規模驗證，不追求完整性）。
    """
    lines = [
        f"- {h['title']} ({h['genres']}) - 評分{h['rating']}" for h in user_history
    ]
    history_block = "\n".join(lines)
    return (
        f"{USER_PROFILE_FEWSHOT}\n\n"
        f"現在請針對以下使用者輸出同樣格式的JSON：\n"
        f"使用者評分紀錄：\n{history_block}\n\n"
        f"輸出："
    )


ITEM_ATTRIBUTE_SYSTEM = (
    "你是電影屬性標註助手。"
    "根據電影片名與既有類型標籤，補充更細緻的風格描述與適合觀眾族群，"
    "目的是補足推薦系統中原始類型標籤過於粗略的問題。"
    "你只能輸出一個JSON物件，禁止輸出任何JSON以外的文字、說明、前言或markdown code block標記（例如```）。"
    "若你不確定這部電影的細節，仍必須依片名與既有類型給出合理推論，"
    "但supplementary_genres若判斷既有類型已足夠完整，可填空陣列[]，不可捏造無關類型。"
    "請全程使用繁體中文回答，絕對不可出現任何簡體字。"
)

ITEM_ATTRIBUTE_FEWSHOT = """範例輸入：
電影：Toy Story
既有類型：Animation, Children's, Comedy

範例輸出：
{"mood_tags": ["溫馨", "幽默", "冒險"], "target_audience": "闔家觀賞", "supplementary_genres": ["Family"], "profile_summary": "以玩具擬人化為主軸的溫馨冒險喜劇，適合親子共賞"}"""


def build_item_attribute_prompt(title: str, genres: str) -> str:
    return (
        f"{ITEM_ATTRIBUTE_FEWSHOT}\n\n"
        f"現在請針對以下電影輸出同樣格式的JSON：\n"
        f"電影：{title}\n既有類型：{genres}\n\n"
        f"輸出："
    )


# 供 parser 檢查用的必要欄位（缺任一欄位即視為格式不穩定，計入失敗率）
USER_PROFILE_REQUIRED_KEYS = {
    "guessed_age_group",
    "preferred_genres",
    "disliked_genres",
    "rating_tendency",
    "profile_summary",
}

ITEM_ATTRIBUTE_REQUIRED_KEYS = {
    "mood_tags",
    "target_audience",
    "supplementary_genres",
    "profile_summary",
}

# 值域檢查（對應 Week3 pilot 發現：required_keys只檢查「有沒有這個欄位」，
# 不檢查「值合不合法」，導致 rating_tendency 實際跑出9種用詞而非設計的3種）。
# key -> 允許的值集合；沒列在這裡的欄位不做值域檢查（自由文字欄位如target_audience/profile_summary）。
USER_PROFILE_ENUM_CHECKS = {
    "rating_tendency": USER_PROFILE_RATING_TENDENCY_VALUES,
    "guessed_age_group": USER_PROFILE_AGE_GROUP_VALUES,
}

ITEM_ATTRIBUTE_ENUM_CHECKS: dict[str, set[str]] = {}
