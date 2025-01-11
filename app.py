from Makima_obj import Makima_class
import time  # Add this import



makima = Makima_class()
def dialogue_with_makima():
    while True:
        # Get user input from speech
        print("я  тебя  слушаю")
        user_input = makima.get_response_from_stt()
        print(f"что я услышала: {user_input}")

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting dialogue with Makima.")
            break

        # Get response from LLM
        llm_response = makima.get_response_from_llm(user_input)
        print(f"Makima: {llm_response['content']}")

        # Convert response to speech and play it
        makima.get_response_from_tts(llm_response['content'])

        # Add a short delay to ensure the user has time to start speaking
        time.sleep(2)

# Start the dialogue
dialogue_with_makima()
