
TG_SOURCES = [
    "https://t.me/datafest",
    "https://t.me/ozon_tech",
    "https://t.me/avitotech",
    "https://t.me/DSCSproAI",
    "https://t.me/Yandex4Developers",
    "https://t.me/kod_zheltyi",
    "https://t.me/freeitevent",
    "https://t.me/iteventsru",
    "https://t.me/it_tus_piter",
    "https://t.me/iteventsrus",
    "https://t.me/ict2go",
    "https://t.me/ict2go_ib",
    "https://t.me/spbit_club",
    "https://t.me/spblug",
    "https://t.me/sbervsochi",
    "https://t.me/itevents_ekb",
    "https://t.me/technohubekb",
    "https://t.me/itmeetupsspb",
    "https://t.me/itsmeetup",
    "https://t.me/codecamp",
    "https://t.me/+btjOZBnV1B03MzNi",
    "https://t.me/Young_and_Yandex",
    "https://t.me/data_secrets",
    "https://t.me/toBeAnMLspecialist",
    "https://t.me/yandexforml",
    "https://t.me/whackdoor",
    "https://t.me/cnewsconf",
    "https://t.me/devrel_rus_offline",
    "https://t.me/forwebdev",
    "https://t.me/front_end_dev",
    "https://t.me/habr_com",
    "https://t.me/ithrconf",
    "https://t.me/ITMeeting",
    "https://t.me/postypashki_old",
    "https://t.me/spbit_club",
    "https://t.me/competech",
    "https://t.me/mmspbu",
    "https://t.me/itmoru",
    "https://t.me/hackathonsrus",
    "https://t.me/hackathons",
    "https://t.me/flux_meetups",
    "https://t.me/it_meeting_ru",
    "https://t.me/afisha4hr",
]

# entity: ссылка, @username или ID канала/группы
# topic_id: ID топика (число). Если это обычный канал или группа целиком, то None.
# limit: ограничение на кол-во сообщений

TARGETS = [
    {
        "name": link.split('/')[-1], 
        "entity": link, 
        "topic_id": None, 
        "limit": None
    } for link in TG_SOURCES
] + [
    {
        "name": "NetworklyAppGroup", 
        "entity": "t.me/NetworklyAppGroup", 
        "topic_id": 689, 
        "limit": None
    }
    ]
