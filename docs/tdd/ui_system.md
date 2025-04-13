## UI System

### Components

#### Panel

This is a technical design for a UI Panel system in Unity with hierarchical structure, optimized for 16:9 aspect ratio in landscape orientation. The panel includes:

- Header Section: Contains back button and title text
- Background: Main content area
- Aspect Ratio: Fixed 16:9 landscape layout

(image)

##### Abstraction

(image)

- **UIPanel**
  - *Hierarchy*:
    - `Header` (parent container)
      - `BackButton` (navigation element)
      - `TitleText` (display text)
    - `Background` (main content area)
  - *Properties*:
    - `aspectRatio`: 16:9 (fixed)
    - `orientation`: Landscape (fixed)
    - `title`: string - Panel title text
    - `showBackButton`: boolean - Back button visibility
  - *Methods*:
    - `SetTitle(string)` - Updates panel title
    - `ToggleBackButton(bool)` - Shows/hides back button
    - `Initialize()` - Sets up default panel state
    - `OnBackPressed()` - Back button click handler

#### Loader (ProgressBar)

This design outlines a loading screen component with progress bar functionality, featuring:

- Background: Full-screen or container-filling element
- Progress Bar: Visual loading indicator
- Configuration: Fixed dimensions (80% width, 30px height)
- Callbacks: Loading progress and completion events

(image)

##### Abstraction

(image)

- **Loader**
  - *Hierarchy*:
    - `Background` (full container element)
    - `ProgressBar` (loading indicator)
  - *Properties*:
    - `width`: 80% of screen width
    - `height`: 30px (fixed)
    - `progress`: float (0-1)
  - *Methods*:
    - `SetProgress(float)` - Updates loading progress
    - `Show()` - Displays the loader
    - `Hide()` - Hides the loader
  - *Callbacks*:
    - `OnLoading` - Called with progress updates
    - `OnFinish` - Called when loading completes

#### Button System

This will be the Base Button abstraction to button-like component

(image)

##### Button System Abstraction

- **BaseButton** (Base Class)
  - *Properties*:
    - `width`: 120 (default)
    - `height`: 40 (default)
    - `buttonText`: string - Display text
    - `isPositive`: boolean - Visual/style flag
  - *Methods*:
    - `Initialize()` - Sets up default button
    - `SetText(string)` - Updates button text
    - `SetSize(width, height)` - Updates dimensions
    - `SetPositive(bool)` - Toggles positive/negative style
  - *Events*:
    - `OnClick` - Click callback


#### Control Button

(image)

##### Abstraction

- **ControlButton** (Extends GeneralButton)
  - *Special Properties*:
    - `isEnabled`: boolean - If button is interactable
    - `countdown`: int - Cooldown timer in seconds
    - `defaultSize`: 100x100 (overrides GeneralButton defaults)
  - *Methods*:
    - `SetEnabled(bool)` - Toggles interactivity
    - `StartCountdown(int)` - Begins cooldown timer
    - `UpdateCountdown()` - Handles timer logic
    - `OnCountdownComplete()` - Called when cooldown finishes
  - *Inherited*:
    - All GeneralButton properties/methods
    - OnClick event

#### Card

(image)

# Abstraction

- **CardMenu** (Extends GeneralButton)
  - *Card Types*:
    - `GameMode`
    - `MultiplayerMode`
    - `ControlMode`
    - `Campaign`
    - `Tutorial`
  - *Properties*:
    - `cardType`: enum - Type of card
    - `overlayText`: string - Additional information display
    - `mapReference`: CampaignMap - Associated map data
    - `isLocked`: bool - If card is interactable
  - *Methods*:
    - `InitializeCard()` - Sets up card visuals
    - `SetOverlayText(string)` - Updates overlay text
    - `SetLockState(bool)` - Toggles interactivity
    - `ShowOverlay()` - Displays additional info
    - `HideOverlay()` - Hides additional info
  - *Inherited*:
    - All GeneralButton properties/methods
    - OnClick event

#### ListView - Vertical Layout Group

(image)

##### Abstraction

###### Core Components
- **Base ListView**
  - *Layout Components*:
    - `VerticalLayoutGroup` - Handles vertical stacking
  - *Configuration*:
    - Child alignment (UpperCenter)
  - *Methods*:
    - `CreateListItem()` - Instantiates list items
    - `Clear()` - Removes all items
  - *Prefabs*:
    - `listItemPrefab` - Base template for all items

###### List Types

###### 1. UnlabeledListView
- *Structure*:
  - Vertical stack of free-form items
  - Optional section headers
- *Features*:
  - Key-value pair display
- *Methods*:
  - `CreateStats()` - Populates with dictionary data
  - `AddHeader()` - Inserts section title
- *Visual*:
  - Center-aligned text
  - Header/bold styling option

###### 2. LabeledListView
- *Structure*:
  - Vertical stack of horizontal rows
  - Column-based layout
- *Features*:
  - Tabular data display
- *Methods*:
  - `CreateTable()` - Builds complete table
  - `AddRow()` - Appends data row
- *Visual*:
  - Center-aligned cells
  - Distinct header row