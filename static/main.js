console.log("aaaaa")

const container = document.getElementById("results");
const form = document.getElementById("prompt-form");

const followupContainer = document.getElementById("follow-up");
const addPromptBtn = document.getElementById("add-prompt-btn");
let promptCount = 1; // start at 1

    addPromptBtn.addEventListener("click", () => {
        promptCount++;

        // wrapper for prompt + remove button
        const wrapper = document.createElement("div");
        wrapper.className = "prompt-wrapper";

        const label = document.createElement("label");
        label.setAttribute("for", `claude-p${promptCount}`);
        label.textContent = `Follow Up Prompt ${promptCount}:`;

        const textarea = document.createElement("textarea");
        textarea.id = `claude-p${promptCount}`;
        textarea.name = "claude-prompts"; // consistent name for all
        textarea.rows = 4;
        textarea.cols = 50;

        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.textContent = "- remove";
        removeBtn.style.marginLeft = "10px";
        removeBtn.addEventListener("click", () => {
            followupContainer.removeChild(wrapper);
        });

        wrapper.appendChild(label);
        wrapper.appendChild(textarea);
        wrapper.appendChild(removeBtn);

        followupContainer.appendChild(wrapper);
    });

form.addEventListener("submit", function(e) {
    e.preventDefault(); // prevent page reload
    container.innerHTML = ""; // clear previous responses

    const mode = form.elements["mode"].value;
    let promptData;

    if (mode === "single") {
        // Single prompt mode: just take the textarea
        promptData = {
            mode: "single",
            model: form.elements["model"].value,
            initial: form.elements["initial-prompt"].value,
            prompt: form.elements["single-prompt"].value
        };
    } else {
        // Multiple prompt mode: gather all inputs
        const followups = Array.from(form.querySelectorAll("textarea[name='claude-prompts']"))
            .map(el => el.value)
            .filter(v => v.trim() !== ""); // skip empty

        promptData = {
            mode: "multiple",
            model: form.elements["model"].value,
            initial: form.elements["initial-prompt"].value,
            prompts: followups
        };
    }

    // Open SSE stream with JSON payload encoded in query string
    const evtSource = new EventSource(
        `/stream?data=${encodeURIComponent(JSON.stringify(promptData))}`
    );

    evtSource.onmessage = function(event) {
        const responses = JSON.parse(event.data);
        container.innerHTML = ""; // replace previous content
        responses.forEach((r, i) => {
            const p = document.createElement("p");
            spk_num = i%2 == 0 ? "1" : "2"
            p.innerHTML = "<b>Speaker " + spk_num + ":</b> " + r;
            container.appendChild(p);
        });
    };

    evtSource.onerror = function() {
        evtSource.close(); // close stream on error
    };
});

const modeSelect = document.getElementById("mode");
const singleSection = document.getElementById("single-prompt-section");
const multipleSection = document.getElementById("multiple-prompts-section");

function toggleSections() {
    if (modeSelect.value === "single") {
        singleSection.style.display = "block";
        multipleSection.style.display = "none";
    } else {
        singleSection.style.display = "none";
        multipleSection.style.display = "block";
    }
}

// Run on page load (in case default selection is not "multiple")
toggleSections();

// Listen for changes
modeSelect.addEventListener("change", toggleSections);