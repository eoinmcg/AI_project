function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'LittleJS Game Tutor 👾';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 20);
            }, i * 50);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);


      const showChat = () => {
        header.style.display = 'none';
        chatbox.style.display = 'block';
        chatinput.style.display = 'block'
        token_counter.style.display = 'none';
      }

      const header = container;
      const input = document.querySelector('.api_key_input');
      input.focus();
      const chatbox = document.querySelector('.chatbox');
      const chatinput = document.querySelectorAll('.form')[1];
      const token_counter = document.querySelector('.token_counter');

    
      chatbox.style.display = 'none';
      chatinput.style.display = 'none';
      token_counter.style.display = 'none';

      input.addEventListener(('blur'), (e) => {
        showChat();
      })
      input.addEventListener(('paste'), (e) => {
        showChat();
      })

    return 'Animation created';
}


