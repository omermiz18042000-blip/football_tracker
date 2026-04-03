Raw Video
   ↓
1. Frame Extraction (read video)
   ↓
2. Object Detection (find players, ball, refs)
   ↓
3. Team Classification (what team is each player on?)
   ↓
4. Player Tracking (assign unique IDs)
   ↓
5. Data Collection (store in DataFrame)
   ↓
6. Video Output + Structured Data

┌─────────────────────────────────────────────────────────┐
│ RAW VIDEO FILE (test_video.mp4)                         │
└──────────────────────┬──────────────────────────────────┘
                       ↓
        ┌──────────────────────────────┐
        │  LOOP: For each frame        │
        └──────────────────────────────┘
                       ↓
      ┌─────────────────────────────────────┐
      │ 1. Extract Frame (cv2.read)         │
      └─────────┬───────────────────────────┘
                ↓
      ┌─────────────────────────────────────┐
      │ 2. Object Detection (YOLO)          │
      │    → Finds bounding boxes           │
      └─────────┬───────────────────────────┘
                ↓
      ┌─────────────────────────────────────┐
      │ 3. Team Classification              │
      │    → Analyzes jersey color          │
      │    → Assigns Red/Blue               │
      └─────────┬───────────────────────────┘
                ↓
      ┌─────────────────────────────────────┐
      │ 4. Player Tracking                  │
      │    → Matches players across frames  │
      │    → Keeps unique IDs               │
      └─────────┬───────────────────────────┘
                ↓
      ┌─────────────────────────────────────┐
      │ 5. Store in Pandas DataFrame        │
      │    (Frame, Player_ID, Team, X, Y)   │
      └─────────┬───────────────────────────┘
                ↓
      ┌─────────────────────────────────────┐
      │ 6. Draw Annotations                 │
      │    (boxes, IDs, colors, ball)       │
      └─────────┬───────────────────────────┘
                ↓
      ┌─────────────────────────────────────┐
      │ 7. Write Frame to Output Video      │
      └─────────┬───────────────────────────┘
                ↓
        ┌──────────────────────────────┐
        │  More frames? YES → Loop     │
        │            NO → Exit         │
        └──────────────────────────────┘
                       ↓
    ┌──────────────────────────────────────┐
    │ OUTPUT:                              │
    │ • Annotated video (processed.mp4)    │
    │ • CSV with all tracking data         │
    │ • Player statistics                  │
    └──────────────────────────────────────┘