let player = document.getElementById("mplayer");
let intervalHandler = null
let hm = document.getElementById("rollContainer")

function _playHandler() {
    player.stop();
    player.currentTime = 0.0;
    player.start();
}

function parseJSON(jsName) {
    let request = new XMLHttpRequest();
    request.open("GET", jsName, false);
    request.send(null)
    return JSON.parse(request.responseText);
}

function playExampleMidi(concept) {
    let conceptPadded = String(concept).padStart(3, '0')
    player.src = `../../assets/midi/concepts/concept_${conceptPadded}.mid`
    player.addEventListener("load", _playHandler);
    hm.removeEventListener("plotly_click", plotCallback)
    clearInterval(intervalHandler)
}

function showInfoPopup() {
    document.getElementById('popupTitle').innerText = name;
    document.getElementById('popup').style.display = 'block';
    document.getElementById('overlay').style.display = 'block';
    document.getElementById('content').classList.add('blurred');
}

function closeInfoPopup() {
    document.getElementById('popup').style.display = 'none';
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('content').classList.remove('blurred');
}

function popSelect(concept, pianist) {
    let jsPath = `../../assets/metadata/cav_sensitivity/${pianist.replace(/ /g, "_").toLowerCase()}.json`
    let jsonData = parseJSON(jsPath);
    let conceptData = jsonData[concept]
    let dropper = document.getElementById("dropdown-menu")
    dropper.innerHTML = "";

    document.getElementById("progressionName").innerHTML = concept;

    for (var i in conceptData) {
        let trackData = conceptData[i];
        let opt = document.createElement('option');
        opt.value = trackData["asset_path"];
        opt.innerHTML = `${trackData["track_name"]} (${trackData["album_name"]}, ${trackData["recording_year"]})`
        dropper.appendChild(opt);
    }
    dropper.value = conceptData[0]["asset_path"]
    dropper.onchange(dropper.value)
}

function setTrack(assetPath) {
    let assetPathNoCavName = assetPath.split('_').slice(0, -2).join('_')
    roll = parseJSON(`../../assets/plots/cav_sensitivity/roll/${assetPathNoCavName}_roll.json`)
    cav = parseJSON(`../../assets/plots/cav_sensitivity/sensitivity/${assetPath}.json`)
    layout = parseJSON(`../../assets/plots/cav_sensitivity/layout/${assetPathNoCavName}_layout.json`)
    layout.shapes = []

    player.removeEventListener("load", _playHandler)
    player.src = `../../assets/midi/cav_sensitivity/${assetPathNoCavName}_midi.mid`

    Plotly.newPlot('rollContainer', [cav, roll], layout);
    hm.on('plotly_click', plotCallback);

    updateLine(0); // Start with the line at y=1
    intervalHandler = setInterval(() => {
        const timeValue = player.currentTime;
        updateLine(timeValue * 100);
    }, 100);
}

function plotCallback(data) {
    console.log(data)
    let pts = '';
    for (var i = 0; i < data.points.length; i++) {
        pts = data.points[i].x / 100
    }
    let currentlyPlaying = player.playing

    player.stop()
    player.currentTime = pts
    updateLine(pts * 100)

    if (currentlyPlaying) {
        player.start()
    }
}

function backToSelection() {
    window.location.href = '../../index.html';
}

function updateLine(xValue) {
    layout.shapes = [{
        type: 'line',
        x0: xValue,
        x1: xValue,
        y0: 0,
        y1: 88,
        line: {
            color: 'yellow',
            width: 2
        }
    }];
    Plotly.react('rollContainer', [cav, roll], layout);
}