"""
Reusable input modal for Gradio applications.
Provides a styled input dialog that matches the Gradio theme.

Usage:
    1. Import INPUT_MODAL_CSS, INPUT_MODAL_HEAD, INPUT_MODAL_HTML
    2. Pass css and head to demo.launch(css=INPUT_MODAL_CSS, head=INPUT_MODAL_HEAD, ...)
    3. Add INPUT_MODAL_HTML via gr.HTML() in your UI
    4. Add a hidden input_trigger textbox with elem_id="input-trigger"
    5. Use show_input_modal_js() for button click handlers
    6. In your callback, process the input: if not value or not value.strip(): return gr.update()

Note: The modal supports pre-filling with a default value (e.g., current selection).
"""

INPUT_MODAL_CSS = """
/* Hide the trigger textbox */
#input-trigger {
  display: none !important;
}

/* Input Modal Styles */
#input-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(4px);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}
#input-modal-overlay.show {
  display: flex;
}
#input-modal-overlay .modal-box {
  background: var(--background-fill-primary);
  border: 1px solid var(--border-color-primary);
  color: var(--body-text-color);
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  min-width: 400px;
  max-width: 500px;
  animation: modalSlideIn 0.2s ease-out;
}
@keyframes modalSlideIn {
  from {
    transform: translateY(-20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}
#input-modal-overlay h3 {
  margin: 0 0 1rem 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--body-text-color);
}
#input-modal-overlay p {
  margin: 0 0 1rem 0;
  color: var(--body-text-color-subdued);
  line-height: 1.5;
  font-size: 0.9rem;
}
#input-modal-overlay input[type="text"] {
  width: 100%;
  padding: 0.625rem;
  border-radius: 6px;
  border: 1px solid var(--border-color-primary);
  background: var(--input-background-fill);
  color: var(--body-text-color);
  font-size: 1rem;
  font-family: inherit;
  margin-bottom: 1.5rem;
  box-sizing: border-box;
}
#input-modal-overlay input[type="text"]:focus {
  outline: none;
  border-color: var(--border-color-accent);
  box-shadow: 0 0 0 3px rgba(255, 165, 0, 0.1);
}
#input-modal-overlay .modal-actions {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
}
#input-modal-overlay .modal-btn {
  padding: 0.625rem 1.5rem;
  border-radius: 6px;
  border: 1px solid var(--border-color-primary);
  background: var(--button-secondary-background-fill);
  color: var(--button-secondary-text-color);
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  font-family: inherit;
  transition: all 0.15s ease;
  height: 42px;
  min-width: 100px;
  line-height: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  box-sizing: border-box;
}
#input-modal-overlay .modal-btn:hover {
  background: var(--button-secondary-background-fill-hover);
  border-color: var(--border-color-accent);
}
#input-modal-overlay .modal-btn.primary {
  background: var(--button-primary-background-fill);
  color: var(--button-primary-text-color);
  border-color: var(--button-primary-border-color);
}
#input-modal-overlay .modal-btn.primary:hover {
  background: var(--button-primary-background-fill-hover);
  border-color: var(--button-primary-border-color-hover);
}
"""

INPUT_MODAL_HEAD = """
<script>
  function submitInputModalValue(action, button) {
    console.log('submitInputModalValue called with action:', action);

    const overlay = document.getElementById('input-modal-overlay');
    const inputField = document.getElementById('input-modal-field');
    
    if (!overlay || !inputField) {
      console.error('Modal elements not found');
      return;
    }

    let valueToSubmit = '';
    
    if (action === 'submit') {
      valueToSubmit = inputField.value.trim();
      console.log('Submitting value:', valueToSubmit);
    } else {
      console.log('Action cancelled');
    }

    overlay.classList.remove('show');
    inputField.value = '';

    // Get context from button's data attribute
    const context = button ? button.getAttribute('data-context') || '' : '';
    const prefixedValue = context + (valueToSubmit || 'cancel');
    console.log('Prefixed value:', prefixedValue);

    // Find the trigger element
    function findTrigger() {
      // Try various selectors for Gradio 5+ compatibility
      let trigger = document.querySelector('#input-trigger textarea');
      if (trigger) return trigger;

      trigger = document.querySelector('#input-trigger input[type="text"]');
      if (trigger) return trigger;

      trigger = document.querySelector('#input-trigger input');
      if (trigger) return trigger;

      const container = document.querySelector('[id="input-trigger"]');
      if (container) {
        trigger = container.querySelector('textarea, input');
        if (trigger) return trigger;
      }

      const allInputs = document.querySelectorAll('textarea, input[type="text"]');
      for (let input of allInputs) {
        const parent = input.closest('[id="input-trigger"]');
        if (parent) return input;
      }

      return null;
    }

    const trigger = findTrigger();

    if (trigger) {
      const newValue = prefixedValue + '_' + Date.now();
      console.log('Setting trigger value to:', newValue);
      trigger.value = newValue;

      trigger.dispatchEvent(new Event('input', { bubbles: true }));
      trigger.dispatchEvent(new Event('change', { bubbles: true }));
      const evt = new InputEvent('input', { bubbles: true, cancelable: true });
      trigger.dispatchEvent(evt);

      console.log('Events dispatched, trigger value is now:', trigger.value);
    } else {
      console.error('Could not find input-trigger element');
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded - setting up input modal listeners');
    const overlay = document.getElementById('input-modal-overlay');
    if (!overlay) {
      console.error('Input modal overlay not found!');
      return;
    }
    
    // Close on overlay click
    overlay.addEventListener('click', function(e) {
      if (e.target === this) {
        submitInputModalValue('cancel');
      }
    });

    // Handle Enter key in input field
    const inputField = document.getElementById('input-modal-field');
    if (inputField) {
      inputField.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
          e.preventDefault();
          const submitBtn = document.getElementById('input-modal-submit-btn');
          if (submitBtn) {
            submitInputModalValue('submit', submitBtn);
          }
        }
      });
    }
  });

  // Close on Escape key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      const overlay = document.getElementById('input-modal-overlay');
      if (overlay && overlay.classList.contains('show')) {
        submitInputModalValue('cancel');
      }
    }
  });
</script>
"""

INPUT_MODAL_HTML = """
<div id="input-modal-overlay">
  <div class="modal-box">
    <h3 id="input-modal-title">Enter Value</h3>
    <p id="input-modal-message">Please enter a value:</p>
    <input type="text" id="input-modal-field" placeholder="Enter text..." />
    <div class="modal-actions">
      <button class="modal-btn" id="input-modal-cancel-btn" onclick="submitInputModalValue('cancel', this)">Cancel</button>
      <button class="modal-btn primary" id="input-modal-submit-btn" onclick="submitInputModalValue('submit', this)">Save</button>
    </div>
  </div>
</div>
"""


def show_input_modal_js(title, message="", placeholder="Enter text...", default_value="", submit_button_text="Save", context=""):
    """
    Generate JavaScript function to show the input modal.

    Args:
        title: Modal title (e.g., "Enter Emotion Name")
        message: Optional message/instruction (e.g., "Enter a name for the preset:")
        placeholder: Placeholder text for input field
        default_value: Default value to pre-fill (can be Gradio variable)
        submit_button_text: Text for the submit button (default: "Save")
        context: Context prefix for the submitted value (e.g., "emotion_", "sample_")

    Returns:
        JavaScript function string to use in Gradio's .click(js=...) parameter
    """
    # If default_value is provided as a string literal, wrap it in quotes
    # Otherwise assume it's a Gradio variable name that will be passed
    return f"""
    (defaultVal) => {{
        const overlay = document.getElementById('input-modal-overlay');
        if (!overlay) return '';
        
        const titleEl = document.getElementById('input-modal-title');
        const messageEl = document.getElementById('input-modal-message');
        const inputField = document.getElementById('input-modal-field');
        const submitBtn = document.getElementById('input-modal-submit-btn');
        const cancelBtn = document.getElementById('input-modal-cancel-btn');

        if (titleEl) titleEl.textContent = {title!r};
        if (messageEl) {{
            messageEl.textContent = {message!r};
            messageEl.style.display = {message!r} ? 'block' : 'none';
        }}
        if (inputField) {{
            inputField.placeholder = {placeholder!r};
            inputField.value = defaultVal || '';
        }}
        if (submitBtn) {{
            submitBtn.textContent = {submit_button_text!r};
            submitBtn.setAttribute('data-context', {context!r});
        }}
        if (cancelBtn) {{
            cancelBtn.setAttribute('data-context', {context!r});
        }}

        overlay.classList.add('show');
        
        // Focus the input field after a brief delay
        setTimeout(() => {{
            if (inputField) {{
                inputField.focus();
                inputField.select();
            }}
        }}, 100);
        
        return '';
    }}
    """
