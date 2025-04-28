
function initPopup(onLoadCallback, elementPrefix) {
    const popupMenuButton = document.getElementById(`${elementPrefix}-popup-button`);
    const popupMenuMenu = document.getElementById(`${elementPrefix}-popup-menu`);
    document.addEventListener('DOMContentLoaded', function () {
        onLoadCallback();
    });
    popupMenuButton.addEventListener('mouseenter', () => {
        popupMenuMenu.style.display = 'block';
    });
    popupMenuButton.addEventListener('mouseleave', () => {
        popupMenuMenu.style.display = 'none';
    });
    popupMenuMenu.addEventListener('mouseenter', () => {
        popupMenuMenu.style.display = 'block';
    });
    popupMenuMenu.addEventListener('mouseleave', () => {
        popupMenuMenu.style.display = 'none';
    });
}

function initErrorPopup() {
    document.getElementById('close-error-popup').addEventListener('click', () => {
        document.getElementById('error-popup').classList.add('hidden');
    });
}

function initSendButtonResize() {
    window.addEventListener('load', updateSendButton);
    window.addEventListener('resize', updateSendButton);
}

function initTextArea(submittedTextList, sendMessageFunc) {
    let submittedTextSelectionCursor = -1;
    let textSubmissionCandidate = '';
    const mainInputTextarea = document.getElementById('input');
    const systemChatMessageInputTextarea = document.getElementById('system-chat-message-input');
    mainInputTextarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
        if (this.style.height < this.style.maxHeight) {
            this.style.overflow = 'hidden';
        } else {
            this.style.overflow = 'auto';
        }
        // Update with each keystroke
        textSubmissionCandidate = mainInputTextarea.value;
    });
    mainInputTextarea.addEventListener('keydown', function(event) {
        if (event.key === 'ArrowUp' && mainInputTextarea.selectionStart === 0) {
            if (submittedTextList.length > 0 && (submittedTextList.length - 1) > submittedTextSelectionCursor) {
                submittedTextSelectionCursor += 1;
                let reversedSubmissions = submittedTextList.slice().reverse();
                mainInputTextarea.value = reversedSubmissions[submittedTextSelectionCursor];
            }
            event.preventDefault();
        }
        if (event.key === 'ArrowDown' && mainInputTextarea.selectionStart === mainInputTextarea.value.length) {
            if (submittedTextList.length > 0 && submittedTextSelectionCursor > -1) {
                submittedTextSelectionCursor -= 1;
                if (submittedTextSelectionCursor === -1) {
                    mainInputTextarea.value = textSubmissionCandidate;
                } else {
                    let reversedSubmissions = submittedTextList.slice().reverse();
                    mainInputTextarea.value = reversedSubmissions[submittedTextSelectionCursor];
                }
            }
            event.preventDefault();
        }
        if (event.key === 'Enter' && (!event.shiftKey)) {
            event.preventDefault();
            sendMessageFunc();

            this.style.height = 'auto';
            this.style.overflow = 'auto';
        }
    });
    window.addEventListener('keydown', (event) => {
        if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
            return;  // Do nothing if keydown is a modifier key.
        }
        if (document.activeElement !== mainInputTextarea && document.activeElement !== systemChatMessageInputTextarea) {
            mainInputTextarea.focus()
        }
    })
}

function getAssistantMessageBox(name, content, iconClass = "fa fa-virus") {
    const assistantMessageBox = document.createElement('div');
    assistantMessageBox.className = 'assistant-message-box';
    assistantMessageBox.innerHTML = `<i class="chat-metadata ${iconClass}"></i><b class="chat-metadata">${name}:</b>`;
    assistantMessageBox.innerHTML += `${marked.parse(content)}</br>`;
    assistantMessageBox.innerHTML += `<i class="chat-metadata">${new Date().toLocaleTimeString()}</i> | `;
    return assistantMessageBox;
}

function getCopyTextLink(text) {
    const textLink = document.createElement('a');
    textLink.title = "Copy";
    textLink.className = "chatbox-message-link";
    textLink.innerHTML = `<i class="fa fa-copy">`;
    textLink.addEventListener('click', (event) => {
        event.preventDefault();
        navigator.clipboard.writeText(text)
            .then(() => {
                console.log('Copied text: ' + text);
            })
            .catch(error => {
                console.log('Failed to copy text:', error)
            });
    });
    return textLink;
}

function getLoadingIndicator() {
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading-indicator';
    loadingIndicator.innerHTML = '<div class="loading-spinner"></div>';
    return loadingIndicator;
}

function getQueryParams() {
    const params = {};
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);

    urlParams.forEach((value, key) => {
        params[key] = value;
    });
    return params;
}

function getUserMessageBox(username, message, codeBlockType = "") {
    const userMessageBox = document.createElement('div');
    userMessageBox.className = 'user-message-box';
    userMessageBox.innerHTML = `<i class="chat-metadata fa fa-user"></i><b class="chat-metadata">${username}:</b>`;
    userMessageBox.innerHTML += `${marked.parse(codeBlockType === '' ? message : `\`\`\`${codeBlockType}\n${message}\n\`\`\``)}</br>`;
    userMessageBox.innerHTML += `<i class="chat-metadata">${new Date().toLocaleTimeString()}</i> | `;
    return userMessageBox;
}

function handleError(error, message) {
    console.error(message, error);
    showErrorPopup(`${message} ${error.message}`);
}

function isBlankOrEmptyStr(str) {
  return !str || str.trim() === '';
}

function scrollChatBoxToBottom() {
    const chatbox = document.getElementById('chatbox');
    chatbox.scrollTop = chatbox.scrollHeight;
}

const showErrorPopup = (message) => {
    const errorPopup = document.getElementById('error-popup');
    const errorMessageText = document.getElementById('error-popup-message-text');

    errorPopup.classList.remove('hidden');
    errorMessageText.innerHTML = message;

    // Auto-hide after 1 minute
    setTimeout(() => {
        errorPopup.classList.add('hidden');
    }, 60000);
};

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Function to check button width and replace text with icon if needed
function updateSendButton() {
    const sendButton = document.getElementById('send-button');
    if (sendButton.offsetWidth < 60) { // Adjust the width as needed
        sendButton.innerHTML = '<i class="fas fa-arrow-right"></i>';
    } else {
        sendButton.innerHTML = 'Send';
    }
}