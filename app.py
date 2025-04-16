# --- Core Flask & DB Imports ---
from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify
from datetime import datetime
# Removed: from db import get_connection # We'll define it here or assume a modified db.py
import os
import logging
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv
from contextlib import contextmanager # For DB connection context manager

# --- PostgreSQL Driver ---
import psycopg2
import psycopg2.extras # Optional: For dictionary cursors if desired

# --- CrewAI / Langchain / Gemini ---
from crewai import Agent, Task, Crew, Process
 # Corrected import path
from langchain_google_genai import GoogleGenerativeAI # Using langchain integration
import google.generativeai as genai # Also keeping direct genai import for analyze_content

# --- Load Environment Variables ---
load_dotenv()

# --- Flask App Setup ---
app = Flask(__name__)
# Use environment variable for secret key, provide a default ONLY for initial local run
app.secret_key = "your_secret_key"
logging.basicConfig(level=logging.INFO)


# --- Configure Gemini ---
# !! Best Practice: Load API Key from Environment Variable !!
google_api_key = os.getenv("GOOGLE_API_KEY") # Use getenv
# google_api_key = "AIzaSyAVn3Y8lugbMGjyQ5AhtD2KvzB-JO6VL9Q" # Avoid hardcoding

if not google_api_key:
    app.logger.error("GOOGLE_API_KEY environment variable not set. AI features may be limited.")
else:
    try:
        genai.configure(api_key=google_api_key)
        app.logger.info("Direct Gemini API configured successfully.")
    except Exception as config_err:
        app.logger.error(f"Error configuring direct Gemini API: {config_err}")

# --- Serper API Key ---
# --- Database Connection Setup (PostgreSQL) ---
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    app.logger.error("FATAL: DATABASE_URL environment variable not set.")
    # Exit or raise error if DB is critical for app start
    # raise ValueError("DATABASE_URL environment variable not set.")

@contextmanager
def get_connection():
    """Provides a PostgreSQL database connection context."""
    conn = None
    cursor = None
    try:
        if not DATABASE_URL:
             raise ValueError("Database connection string not configured.")
        conn = psycopg2.connect(DATABASE_URL)
        # Use DictCursor for easy column access by name (optional)
        # Requires psycopg2[extras] or psycopg2-binary[extras]
        # cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor = conn.cursor() # Standard cursor
        app.logger.debug("Database connection obtained.")
        yield conn, cursor # Yield both
        conn.commit()
        app.logger.debug("Transaction committed.")
    except (Exception, psycopg2.DatabaseError) as error:
        app.logger.error(f"Database Error: {error}")
        if conn:
            conn.rollback()
            app.logger.warning("Transaction rolled back due to error.")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            app.logger.debug("Database connection closed.")

# --- Context Processor (Using new DB connection) ---
@app.context_processor
def inject_unread_count():
    unread_count = 0
    if 'user_id' in session:
        try:
            # Use %s for PostgreSQL placeholders, ensure table/column names match PG schema (quoted if needed)
            with get_connection() as (conn, cursor):
                cursor.execute('SELECT COUNT(*) FROM "Notifications" WHERE recipient_user_id=%s AND is_read=FALSE',
                               (session['user_id'],)) # Note trailing comma for single param tuple
                result = cursor.fetchone()
                if result:
                    unread_count = result[0]
        except Exception as e:
             app.logger.error(f"Error fetching unread count: {e}")
    # Also pass friend request count if needed by layout.html directly
    friend_requests_count = 0
    if 'user_id' in session:
        try:
             with get_connection() as (conn, cursor):
                cursor.execute('SELECT COUNT(*) FROM "FriendRequests" WHERE receiver_id = %s AND status = %s',
                               (session['user_id'], 'pending'))
                result = cursor.fetchone()
                if result:
                    friend_requests_count = result[0]
        except Exception as e:
             app.logger.error(f"Error fetching friend request count: {e}")

    return {'unread_count': unread_count, 'friend_requests_count': friend_requests_count}


# --- App Config ---
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['MAX_ABUSE_COUNT'] = 3

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the filename has an allowed image extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Chatbot Initialization and Handling ---
# (Keep the functions: initialize_chatbot_agents, needs_search,
# handle_search_query, handle_conversation, clean_response from the previous answers)
# Ensure they use app.logger and check for API keys correctly.
# --- START: Chatbot Functions ---
_agents_cache = None
def initialize_chatbot_agents():
    global _agents_cache
    if _agents_cache: return _agents_cache
    app.logger.info("Initializing chatbot agents...")
    search_tool = None
    llm_args = {"google_api_key": google_api_key} if google_api_key else {}
    if not llm_args: app.logger.error("Cannot initialize LLMs without GOOGLE_API_KEY."); return None, None, None
    try:
        llm_instance = GoogleGenerativeAI(model="gemini-1.5-flash", **llm_args) # Using Langchain Google integration
        app.logger.info(f"LLM Instance ({llm_instance.model}) created.")
        conversation_agent = Agent(role="Conversation Assistant", goal="Engage in helpful and informative conversations.", backstory="A friendly AI assistant.", llm=llm_instance, verbose=False, allow_delegation=False)
        search_agent_tools = [search_tool] if search_tool else []
        search_agent = Agent(role="Information Retrieval Specialist", goal="Find relevant and up-to-date information online.", backstory="An expert researcher using web search.", llm=llm_instance, tools=search_agent_tools, verbose=False, allow_delegation=False)
        writer_agent = Agent(role="Content Editor and Summarizer", goal="Refine and condense information into concise, user-friendly responses.", backstory="An expert editor simplifying info.", llm=llm_instance, verbose=False, allow_delegation=False)
        _agents_cache = (conversation_agent, search_agent, writer_agent)
        return _agents_cache
    except Exception as e: app.logger.error(f"Error initializing Agents/LLMs: {e}"); return None, None, None
def needs_search(query):
    query_lower = query.lower().strip(); # ... (keep rest of your logic) ...
    search_keywords = ['latest', 'news', 'recent', 'update', 'current', 'today', 'yesterday', 'this week', 'this month', 'stock', 'weather', 'price', 'event', 'score', 'happening', 'announced', 'released', 'published', 'launched', 'define', 'what is', 'who is', 'where is', 'when is', 'how to', 'why is', 'compare', 'statistics on', 'population of', 'capital of']
    time_indicators = ['now', 'currently', 'latest', 'recent', 'update', 'new', 'just', 'breaking', 'live']
    if not query_lower: return False
    if query_lower.startswith('search for') or query_lower.startswith('find information on'): return True
    if any(keyword in query_lower for keyword in search_keywords): return True
    if any(indicator in query_lower for indicator in time_indicators): return True
    question_starters = ['what', 'who', 'where', 'when', 'why', 'how']
    if any(query_lower.startswith(q) for q in question_starters) and len(query.split()) > 3:
        if query_lower not in ["how are you", "how are you doing", "how's it going"]: return True
    if len(query.split()) > 5 and ('?' in query or query_lower.startswith('tell me about')):
        if not any(phrase in query_lower for phrase in ['what do you think', 'your opinion on', 'do you believe', 'how do you feel']): return True
    app.logger.debug(f"Query '{query[:50]}...' determined NOT to need search.")
    return False
def clean_response(response):
    if not isinstance(response, str): app.logger.warning(f"Received non-string response: {type(response)}"); return "Sorry, unexpected response format."
    lines = response.splitlines()
    cleaned_lines = [line for line in lines if not any(line.strip().lower().startswith(kw) for kw in ["task output:", "final answer:", "action:", "action input:", "observation:", "thought:", "***", "tool response:"])]
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = ' '.join(cleaned.split()); cleaned = cleaned.replace('**', '')
    return cleaned if cleaned else "Sorry, I couldn't generate a suitable response."
def handle_search_query(query):
    app.logger.info(f"Handling search query for: '{query[:50]}...'"); agents = initialize_chatbot_agents()
    if not all(agents): return "Sorry, chatbot components failed to initialize."
    _, search_agent, writer_agent = agents
    if not search_agent.tools: app.logger.warning("Search query, but search tool unavailable. Fallback."); return handle_conversation(f"User asked '{query}', but web search is unavailable. Respond based on general knowledge or state inability.")
    search_task = Task(description=f"Perform web search for relevant, accurate, up-to-date info for query: '{query}'. Synthesize key findings.", agent=search_agent, expected_output="Concise summary of key facts/info from search directly answering the query.")
    writer_task = Task(description=f"Review search summary from context. Rewrite into clear, concise, natural conversational response for user who asked: '{query}'. Focus on search context ONLY. Aim for 2-4 sentences.", agent=writer_agent, expected_output="Polished, user-friendly paragraph summarizing search findings.", context=[search_task])
    crew = Crew(agents=[search_agent, writer_agent], tasks=[search_task, writer_task], verbose=False, process=Process.sequential)
    try: result = crew.kickoff(inputs={'query': query}); app.logger.info(f"Search crew finished."); return clean_response(result)
    except Exception as e: app.logger.error(f"Search crew error: {e}", exc_info=True); return "Sorry, error during search."
def handle_conversation(query):
    app.logger.info(f"Handling conversation query for: '{query[:50]}...'"); agents = initialize_chatbot_agents()
    if not all(agents): return "Sorry, chatbot components failed to initialize."
    conversation_agent, _, writer_agent = agents
    conversation_task = Task(description=f"Engage with user message: '{query}'. Respond helpfully, conversationally based on general knowledge. If unsure, say so politely.", agent=conversation_agent, expected_output="Relevant, polite, conversational response paragraph.")
    writer_task = Task(description=f"Review conversational response from context. Ensure concise, clear, addresses: '{query}'. Refine for flow. Aim 2-4 sentences.", agent=writer_agent, expected_output="Polished, concise, user-friendly conversational response.", context=[conversation_task])
    crew = Crew(agents=[conversation_agent, writer_agent], tasks=[conversation_task, writer_task], verbose=False, process=Process.sequential)
    try: result = crew.kickoff(); app.logger.info(f"Conversation crew finished."); return clean_response(result)
    except Exception as e: app.logger.error(f"Conversation crew error: {e}", exc_info=True); return "Sorry, error processing message."
# --- END: Chatbot Functions ---

# --- Direct Gemini Analysis Function (kept separate from chatbot logic) ---
def analyze_content_with_gemini(text_content, image_path=None):
    """
    Analyzes text and optionally an image using Gemini for safety/positivity.
    Returns True if deemed negative/unsafe, False otherwise or if API fails/unavailable.
    """
    api_key_present = bool(os.environ.get("GOOGLE_API_KEY"))
    if not api_key_present:
        app.logger.warning("Skipping Gemini analysis: API key not configured.")
        return False # Default to safe if API is unavailable

    try:
        # Prepare input parts (text is always present, image is optional)
        input_parts = []
        log_image_info = "None"

        # --- CHOOSE and REFINE your prompt ---
        # Determine if it's text-only or multimodal
        if image_path and os.path.exists(image_path):
            analysis_type = "Multimodal (Text + Image)"
            img = Image.open(image_path)
            input_parts.append(img) # Add image first if present
            log_image_info = os.path.basename(image_path)
            # Prompt for combined analysis
            prompt = f"""
            Analyze the sentiment and safety of the provided image and text comment based on standard community guidelines
            (avoiding hate speech, explicit content, excessive violence, harassment, illegal acts).
            Is the overall message and visual POSITIVE or NEGATIVE/UNSAFE?
            Respond strictly with the single word 'POSITIVE' or 'NEGATIVE'.

            Text Comment (provide context even if empty): "{text_content}"
            """
        elif text_content:
            analysis_type = "Text-Only"
            # Prompt for text-only analysis
            prompt = f"""
            Analyze the sentiment and safety of the following text based on standard community guidelines
            (avoiding hate speech, explicit content, excessive violence, harassment, illegal acts).
            Is the text POSITIVE or NEGATIVE/UNSAFE?
            Respond strictly with the single word 'POSITIVE' or 'NEGATIVE'.

            Text: "{text_content}"
            """
        else:
            # Should not happen if validation in routes is correct, but handle defensively
            app.logger.warning("analyze_content_with_gemini called with no text or image.")
            return False # No content to analyze

        input_parts.insert(0, prompt) # Add prompt as the first part

        app.logger.debug(f"Analyzing with Gemini ({analysis_type}) - Text: '{text_content[:50]}...', Image: {log_image_info}")

        # Select the appropriate Gemini model
        # Use a model that supports the input type (text-only or multimodal)
        # 'gemini-1.5-pro' should handle both, 'gemini-pro' for text-only if needed
        model_name = 'gemini-1.5-pro' if image_path else 'gemini-1.5-pro' # Or use gemini-pro for text
        model = genai.GenerativeModel(model_name)

        # Generate content
        response = model.generate_content(input_parts)

        # Basic safety check on response
        if not response.parts:
            app.logger.warning("Gemini response has no parts. Treating as inconclusive/safe.")
            return False
        # Check for blocks due to prompt or response safety filters
        if response.prompt_feedback.block_reason:
            app.logger.warning(f"Gemini request blocked for prompt. Reason: {response.prompt_feedback.block_reason}. Treating as potentially unsafe.")
            return True # Treat blocked prompts as unsafe
        # You might also check response.candidates[0].finish_reason == 'SAFETY'
        if response.candidates and response.candidates[0].finish_reason == 'SAFETY':
             app.logger.warning(f"Gemini response blocked for safety. Treating as unsafe.")
             return True # Treat blocked responses as unsafe


        # --- Parse the Response ---
        response_text = response.text.strip().upper()
        app.logger.info(f"Gemini Raw Response: '{response.text}', Parsed Classification: '{response_text}'")

        # Check against your chosen prompt's expected output (POSITIVE/NEGATIVE)
        if response_text == 'NEGATIVE':
            app.logger.info(f"Gemini analysis result: NEGATIVE (Flagged as Abusive: True)")
            return True # Negative -> considered abusive
        elif response_text == 'POSITIVE':
            app.logger.info(f"Gemini analysis result: POSITIVE (Flagged as Abusive: False)")
            return False # Positive -> not abusive
        else:
             app.logger.warning(f"Gemini returned unexpected classification: '{response.text}'. Defaulting to 'safe/positive' (Abusive: False).")
             return False # Default to safe if classification is unclear

    except FileNotFoundError:
        app.logger.error(f"Gemini Analysis Error: Image file not found at path {image_path}")
        return False
    except Exception as e:
        # Catch potential API errors, PIL errors, etc.
        app.logger.error(f"Error during Gemini analysis or processing its response: {str(e)}")
        return False
# --- Core Application Routes (Modified for PostgreSQL) ---

@app.route('/')
def home():
    # If user logged in, maybe redirect to dashboard?
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html') # Render landing page if not logged in

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'] # !! COMPARE HASHED PASSWORDS !!

        try:
            with get_connection() as (conn, cursor):
                # Use double quotes for table/column names if needed by your schema
                cursor.execute('SELECT id, username, password, status FROM "Users" WHERE username=%s', (username,))
                user = cursor.fetchone()

                # !! IMPORTANT: Replace plain text check with hash verification !!
                # Example using werkzeug.security (install it: pip install Werkzeug)
                # from werkzeug.security import check_password_hash
                # stored_hash = user[2] if user else None
                # if user and check_password_hash(stored_hash, password):
                if user and user[2] == password: # TEMPORARY - REPLACE WITH HASH CHECK
                    if user[3] == 'active':
                        session['user_id'] = user[0]
                        session['username'] = user[1]
                        flash("Login successful!", "success")
                        return redirect(url_for('dashboard'))
                    else:
                        flash("Your account has been deactivated.", "danger")
                else:
                    flash("Invalid username or password.", "danger")
        except Exception as e:
            app.logger.error(f"Login error: {e}", exc_info=True)
            flash("An error occurred during login.", "error")
            # Don't redirect on error, show login page again
            return render_template('auth.html', mode='login')

    # GET Request
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('auth.html', mode='login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password')

        if not username or not password:
             flash("Username and password are required.", "warning"); return render_template('auth.html', mode='register')
        if password != confirm_password:
             flash("Passwords do not match.", "warning"); return render_template('auth.html', mode='register')
        if len(password) < 6: # Basic password length check
             flash("Password must be at least 6 characters long.", "warning"); return render_template('auth.html', mode='register')

        try:
            with get_connection() as (conn, cursor):
                # Check if username exists
                cursor.execute('SELECT id FROM "Users" WHERE username=%s', (username,))
                if cursor.fetchone():
                    flash("Username already exists.", "warning")
                    return render_template('auth.html', mode='register')

                # !! IMPORTANT: HASH the password before storing !!
                # from werkzeug.security import generate_password_hash
                # hashed_password = generate_password_hash(password)
                hashed_password = password # TEMPORARY - REPLACE WITH HASHING

                cursor.execute(
                    'INSERT INTO "Users" (username, password, abuse_count, status) VALUES (%s, %s, 0, %s)',
                    (username, hashed_password, 'active')
                )
                # Commit is handled by context manager

            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))

        except psycopg2.IntegrityError as ie: # Catch specific unique violation
             app.logger.warning(f"Registration failed (IntegrityError): {ie}")
             flash("Username already exists.", "warning")
             return render_template('auth.html', mode='register')
        except Exception as e:
            app.logger.error(f"Registration error: {e}", exc_info=True)
            flash("An error occurred during registration.", "error")
            return render_template('auth.html', mode='register')

    # GET Request
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('auth.html', mode='register')


@app.route('/logout')
def logout():
     session.clear()
     flash("You have been logged out.", "info")
     return redirect(url_for('home')) # Redirect to landing page after logout

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    current_user_id = session['user_id']
    posts_with_details = []

    try:
        with get_connection() as (conn, cursor):
            # Fetch posts with username
            cursor.execute("""
                SELECT p.id, p.user_id, p.content, p.media_url, p.media_type, p.created_at, u.username
                FROM "Posts" p JOIN "Users" u ON u.id = p.user_id
                ORDER BY p.created_at DESC
            """)
            posts_raw = cursor.fetchall()

            # Create a list of post IDs to fetch comments/reactions efficiently
            post_ids = [p[0] for p in posts_raw]

            # Fetch all comments for these posts in one go
            comments_by_post = {}
            if post_ids: # Only query if there are posts
                cursor.execute("""
                    SELECT c.id, c.post_id, c.user_id, c.content, c.media_url, c.media_type, c.created_at, u.username
                    FROM "Comments" c JOIN "Users" u ON u.id = c.user_id
                    WHERE c.post_id = ANY(%s) ORDER BY c.post_id, c.created_at ASC
                """, (post_ids,)) # Use ANY for list query in PG
                comments_raw = cursor.fetchall()
                for c in comments_raw:
                    post_id = c[1]
                    if post_id not in comments_by_post:
                        comments_by_post[post_id] = []
                    comments_by_post[post_id].append({
                        'id': c[0], 'post_id': c[1], 'user_id': c[2], 'content': c[3],
                        'media_url': c[4], 'media_type': c[5], 'created_at': c[6], 'username': c[7]
                    })

            # Fetch all reactions for these posts (counts and user's)
            reactions_by_post = {}
            user_reactions_by_post = {}
            if post_ids:
                 # Get counts
                 cursor.execute("""
                     SELECT post_id, reaction_type, COUNT(*) as count
                     FROM "Reactions"
                     WHERE post_id = ANY(%s)
                     GROUP BY post_id, reaction_type
                 """, (post_ids,))
                 reactions_raw = cursor.fetchall()
                 for r in reactions_raw:
                     post_id, r_type, count = r
                     if post_id not in reactions_by_post: reactions_by_post[post_id] = {}
                     reactions_by_post[post_id][r_type] = count

                 # Get current user's reaction
                 cursor.execute("""
                    SELECT post_id, reaction_type FROM "Reactions"
                    WHERE user_id = %s AND post_id = ANY(%s)
                 """, (current_user_id, post_ids))
                 user_reactions_raw = cursor.fetchall()
                 for r in user_reactions_raw:
                     user_reactions_by_post[r[0]] = r[1]

            # Assemble the final data structure
            for p in posts_raw:
                post_id = p[0]
                post_reactions = reactions_by_post.get(post_id, {})
                total_reactions = sum(post_reactions.values())
                user_reaction = user_reactions_by_post.get(post_id, None)

                post_dict = {
                    'id': post_id,
                    'user_id': p[1],
                    'content': p[2],
                    'media_url': p[3],
                    'media_type': p[4],
                    'created_at': p[5],
                    'username': p[6],
                    'comments': comments_by_post.get(post_id, []),
                    'reactions': post_reactions,
                    'total_reactions': total_reactions,
                    'current_user_reaction': user_reaction
                }
                posts_with_details.append(post_dict)

    except Exception as e:
        app.logger.exception(f"Error loading dashboard: {e}")
        flash("Error loading dashboard content.", "error")

    return render_template('dashboard.html', posts=posts_with_details)


@app.route('/post', methods=['GET', 'POST'])
def create_post():
    if 'user_id' not in session: return redirect(url_for('login'))

    if request.method == 'POST':
        content = request.form.get('content', '').strip()
        user_id = session['user_id']
        username = session.get('username', 'User') # Get username for logging/flash

        media_url = None
        media_type = None
        filepath = None
        file = request.files.get('media')

        # --- File Upload Logic ---
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                # Consider adding user_id or timestamp to filename to prevent collisions
                # filename = f"{user_id}_{int(datetime.now().timestamp())}_{secure_filename(file.filename)}"
                upload_dir = app.config['UPLOAD_FOLDER']
                os.makedirs(upload_dir, exist_ok=True)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                media_url = f"uploads/{filename}" # Relative path for URL
                # Simple check for image/video based on extension (improve if needed)
                ext = filename.rsplit('.', 1)[1].lower()
                if ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']:
                    media_type = 'image'
                elif ext in ['mp4', 'webm', 'mov', 'avi', 'ogg']: # Add more video types if needed
                     media_type = 'video'
                else:
                     media_type = 'other' # Or reject unsupported types
                app.logger.debug(f"Uploaded {media_type} for post: {media_url}")
            except Exception as e:
                app.logger.error(f"Error uploading file for post by user {user_id}: {e}", exc_info=True)
                flash("Error uploading file.", "error")
                filepath = None; media_url = None; media_type = None # Reset on error

        # Validation
        if not content and not media_url:
            flash("Post cannot be empty. Please provide text or media.", "warning")
            return render_template('post.html') # Stay on page

        # --- Content Analysis ---
        image_path_for_analysis = filepath if media_type == 'image' else None
        is_overall_abusive = analyze_content_with_gemini(content, image_path_for_analysis)
        app.logger.info(f"Gemini analysis for post: negative/unsafe={is_overall_abusive}")

        # --- Database Operations ---
        try:
            with get_connection() as (conn, cursor):
                # --- Handle Abuse ---
                if is_overall_abusive:
                    app.logger.info(f"Abusive post detected by user {user_id}")
                    cursor.execute('SELECT abuse_count FROM "Users" WHERE id = %s', (user_id,))
                    count_result = cursor.fetchone()
                    if count_result:
                        new_count = count_result[0] + 1
                        app.logger.debug(f"Updating abuse count for user {user_id} to {new_count}")
                        cursor.execute('UPDATE "Users" SET abuse_count = %s WHERE id = %s', (new_count, user_id))
                        if new_count >= app.config['MAX_ABUSE_COUNT']:
                            app.logger.warning(f"Deactivating user {user_id} ({username}) due to abuse count.")
                            cursor.execute('UPDATE "Users" SET status = %s WHERE id = %s', ('deactivated', user_id))
                            # Commit needed before clearing session & redirecting
                            conn.commit()
                            session.clear()
                            flash("Your account has been deactivated due to repeated content violations.", "danger")
                            return redirect(url_for('login'))
                        else:
                            warnings_left = app.config['MAX_ABUSE_COUNT'] - new_count
                            flash(f"Warning: Your post may violate guidelines. {warnings_left} warnings remaining.", "warning")
                    else: app.logger.error(f"Could not find user {user_id} for abuse count update.")
                    # Decide if abusive posts should still be inserted or not. Here we continue to insert.

                # --- Insert Post ---
                sql_insert_post = """
                    INSERT INTO "Posts" (user_id, content, media_url, media_type, created_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    RETURNING id;
                """
                cursor.execute(sql_insert_post, (user_id, content, media_url, media_type))
                post_id_result = cursor.fetchone()
                if post_id_result and post_id_result[0]:
                    post_id = post_id_result[0]
                    app.logger.info(f"Inserted post with ID: {post_id}")
                    # Commit is handled by context manager on successful exit
                    if not is_overall_abusive:
                        flash("Post created successfully!", "success")
                    return redirect(url_for('dashboard'))
                else:
                    app.logger.error("CRITICAL: Failed to retrieve post ID using RETURNING clause after insert.")
                    raise Exception("Failed to retrieve post ID after insert.")

        except Exception as e:
            # Rollback is handled by context manager on exception
            flash("Error saving post. Please try again.", "error")
            app.logger.exception(f"Unhandled error creating post by user {user_id}: {e}")
            # If file was uploaded but DB failed, maybe delete the file?
            # if filepath and os.path.exists(filepath): os.remove(filepath)
            return render_template('post.html') # Stay on post page on error

    # GET request
    return render_template('post.html')


@app.route('/comment/<int:post_id>', methods=['POST'])
def comment(post_id):
    if 'user_id' not in session:
        flash("Please login to comment.", "warning")
        return redirect(url_for('login'))

    content = request.form.get('content', '').strip()  # Typed text
    user_id = session['user_id']
    username = session.get('username', 'A user')

    # --- File Upload ---
    media_url = None
    media_type = None
    filepath = None
    file = request.files.get('media')
    if file and allowed_file(file.filename):  # Checks for allowed image types
        try:
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            media_url = f"uploads/{filename}"
            media_type = 'image'
            app.logger.debug(f"Uploaded image for comment: {media_url}")
        except Exception as e:
            app.logger.error(f"Error uploading image for comment on post {post_id}: {e}")
            flash("Error uploading image.", "error")
            filepath = None; media_url = None; media_type = None

    # Validation
    if not content and not media_url:
        flash("Comment cannot be empty. Please provide text or an image.", "warning")
        return redirect(url_for('dashboard'))

    # --- Content Analysis with Gemini ---
    image_path_for_analysis = filepath if media_type == 'image' else None
    is_overall_abusive = analyze_content_with_gemini(content, image_path_for_analysis)
    analysis_method = "Gemini" if image_path_for_analysis else "Gemini (Text Only)"
    app.logger.info(f"Gemini analysis result for comment (negative/unsafe={is_overall_abusive}) using {analysis_method}")

    # --- Database Operations ---
    with get_connection() as (conn, cursor):
        comment_id = None  # Initialize
        try:
            # Get Post Owner
            cursor.execute('SELECT user_id FROM "Posts" WHERE id = %s', (post_id,))
            post_owner_result = cursor.fetchone()
            if not post_owner_result:
                flash("Post not found.", "error")
                return redirect(url_for('dashboard'))
            post_owner_id = post_owner_result[0]

            # Insert Comment & Get ID
            sql_insert_comment = """
                INSERT INTO "Comments" (post_id, user_id, content, media_url, media_type, created_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING id;
            """
            try:
                cursor.execute(sql_insert_comment, (post_id, user_id, content, media_url, media_type))
                identity_result = cursor.fetchone()
                if identity_result and identity_result[0] is not None:
                    comment_id = identity_result[0]
                    app.logger.info(f"Successfully extracted comment_id using RETURNING: {comment_id}")
                else:
                    app.logger.error(f"CRITICAL: Failed to retrieve comment ID using RETURNING clause.")
                    raise Exception("Failed to retrieve comment ID after insert.")  # Raise to trigger rollback
            except Exception as db_err:
                app.logger.error(f"DB error during comment INSERT/RETURNING: {db_err}")
                raise  # Re-raise to trigger rollback

            # --- Handle Abuse (if flagged by Gemini) ---
            if is_overall_abusive:
                app.logger.info(f"Abusive content detected (Comment ID: {comment_id}, Method: {analysis_method}) on post {post_id} by user {user_id}")
                # Update abuse count logic...
                cursor.execute('SELECT abuse_count FROM "Users" WHERE id = %s', (user_id,))
                count_result = cursor.fetchone()
                if count_result:
                    current_count = count_result[0]
                    new_count = current_count + 1
                    app.logger.debug(f"Updating abuse count for user {user_id} from {current_count} to {new_count}")
                    cursor.execute('UPDATE "Users" SET abuse_count = %s WHERE id = %s', (new_count, user_id))

                    # Check for deactivation...
                    if new_count >= app.config['MAX_ABUSE_COUNT']:
                        app.logger.warning(f"Deactivating user {user_id} ({username}) due to abuse count.")
                        # Deactivation needs commit before redirect
                        cursor.execute('UPDATE "Users" SET status = %s WHERE id = %s', ('deactivated', user_id))
                        conn.commit()
                        session.clear()
                        flash("Your account has been deactivated...", "danger")
                        return redirect(url_for('login'))
                    else:
                        warnings_left = app.config['MAX_ABUSE_COUNT'] - new_count
                        flash(f"Warning: Your comment may violate guidelines. {warnings_left} warnings remaining.", "warning")
                else:
                    app.logger.error(f"Could not find user {user_id} to update abuse count.")

                # --- Send Notification (if abusive, on someone else's post) ---
                if post_owner_id != user_id:
                    # comment_id is guaranteed non-None if we reached here
                    cursor.execute('SELECT username FROM "Users" WHERE id = %s', (user_id,))
                    sender_result = cursor.fetchone()
                    sender_username = sender_result[0] if sender_result else f"User {user_id}"

                    app.logger.info(f"Creating notification for post owner {post_owner_id} about abusive comment {comment_id} from {sender_username}...")
                    try:
                        cursor.execute("""
                            INSERT INTO "Notifications" (recipient_user_id, sender_user_id, message, content_type, content_id, is_read, created_at)
                            VALUES (%s, %s, %s, %s, %s, FALSE, CURRENT_TIMESTAMP)
                            """,
                            (post_owner_id, user_id,
                             f"User {sender_username} posted a comment that may violate guidelines (check content/image)",
                             'comment', comment_id)
                        )
                        app.logger.debug("Executed INSERT INTO Notifications")
                    except Exception as notify_err:  # Catch specific DB or other errors
                        app.logger.error(f"Error inserting notification for comment {comment_id}: {notify_err}")
                        flash("Comment added, but failed to send notification.", "warning")
                else:
                    app.logger.debug(f"Abusive comment {comment_id} was on user's own post. No notification needed.")
            # --- End of Abuse Handling ---

            # Flash success only if not abusive
            if not is_overall_abusive:
                flash("Comment added successfully!", "success")

        except Exception as e:
            # General catch-all - rollback will happen automatically due to 'with' block exit on exception
            flash("Error processing comment. Please try again.", "error")
            app.logger.exception(f"Unhandled error adding comment by user {user_id} on post {post_id}: {str(e)}")

    return redirect(url_for('dashboard'))


# --- Notification Routes (Modified for PostgreSQL) ---
@app.route('/notifications')
def notifications():
    if 'user_id' not in session: return redirect(url_for('login'))
    user_id = session['user_id']
    notifications_list = []
    try:
        with get_connection() as (conn, cursor):
            # Use LEFT JOIN in case sender was deleted but notification kept
            cursor.execute("""
                SELECT n.id, n.recipient_user_id, n.sender_user_id, u.username as sender_username,
                       n.message, n.is_read, n.content_type, n.content_id, n.created_at
                FROM "Notifications" n LEFT JOIN "Users" u ON n.sender_user_id = u.id
                WHERE n.recipient_user_id = %s
                ORDER BY n.created_at DESC
            """, (user_id,))
            raw_notifications = cursor.fetchall()

            # Convert rows to dictionaries (if not using DictCursor)
            notifications_list = [{
                'id': row[0], 'recipient_user_id': row[1], 'sender_user_id': row[2],
                'sender_username': row[3] or "System", 'message': row[4], 'is_read': row[5],
                'content_type': row[6], 'content_id': row[7], 'created_at': row[8]
            } for row in raw_notifications if row[0] is not None]

            # Mark as read only AFTER fetching
            if notifications_list:
                 cursor.execute('UPDATE "Notifications" SET is_read = TRUE WHERE recipient_user_id = %s AND is_read = FALSE', (user_id,))
                 # Commit handled by context manager

    except Exception as e:
        flash("Error loading notifications.", "error")
        app.logger.error(f"Notifications loading error: {e}", exc_info=True)

    return render_template('notifications.html', notifications=notifications_list)

@app.route('/approve_content/<int:notification_id>', methods=['POST'])
def approve_content(notification_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    try:
        with get_connection() as (conn, cursor):
            # Verify ownership first
            cursor.execute('SELECT id FROM "Notifications" WHERE id = %s AND recipient_user_id = %s',
                           (notification_id, session['user_id']))
            if not cursor.fetchone():
                flash("Notification not found or not authorized.", "error"); return redirect(url_for('notifications'))

            # Delete the notification
            cursor.execute('DELETE FROM "Notifications" WHERE id = %s', (notification_id,))
            rows_deleted = cursor.rowcount # psycopg2 cursor has rowcount
            # Commit handled by context manager

            if rows_deleted > 0: flash("Content approved (notification cleared).", "success")
            else: flash("Notification could not be cleared.", "warning")
    except Exception as e:
        flash("Error approving content.", "error")
        app.logger.exception(f"Approval error notification {notification_id}: {e}")
    return redirect(url_for('notifications'))

@app.route('/remove_content/<int:notification_id>', methods=['POST'])
def remove_content(notification_id):
     if 'user_id' not in session: return redirect(url_for('login'))
     try:
        with get_connection() as (conn, cursor):
            # Get notification details and verify ownership
            cursor.execute('SELECT n.id, n.content_type, n.content_id FROM "Notifications" n WHERE n.id = %s AND n.recipient_user_id = %s',
                           (notification_id, session['user_id']))
            result = cursor.fetchone()
            if not result:
                flash("Notification not found or not authorized.", "error"); return redirect(url_for('notifications'))

            notif_id, content_type, content_id = result
            app.logger.info(f"Attempting removal via Notif ID: {notif_id}, type: {content_type}, content_id: {content_id}.")

            rows_deleted_content = 0
            if content_type == 'comment' and content_id:
                 # Make sure user is owner of the *post* the comment is on, or owner of the comment itself?
                 # For now, assume recipient of notification (post owner) can delete comment
                 cursor.execute('DELETE FROM "Comments" WHERE id = %s', (content_id,))
                 rows_deleted_content = cursor.rowcount
                 app.logger.info(f"Deleted comment {content_id}. Rows affected: {rows_deleted_content}")
            elif content_type == 'post' and content_id:
                 # Add logic to delete post if needed - requires check if user_id matches post.user_id
                 # cursor.execute('DELETE FROM "Posts" WHERE id = %s AND user_id = %s', (content_id, session['user_id'])) etc.
                 app.logger.warning(f"Post removal via notification not fully implemented yet.")
                 pass
            else:
                app.logger.warning(f"Notification {notif_id} lacks valid content_type/content_id for removal.")

            # Always delete the notification itself
            cursor.execute('DELETE FROM "Notifications" WHERE id = %s', (notif_id,))
            rows_deleted_notification = cursor.rowcount
            # Commit handled by context manager

            # Feedback logic...
            if content_id and rows_deleted_content > 0 and rows_deleted_notification > 0: flash(f"{content_type.capitalize()} and notification removed.", "success")
            elif content_id and rows_deleted_content == 0 and rows_deleted_notification > 0: flash(f"{content_type.capitalize()} may have already been removed; notification cleared.", "warning")
            elif not content_id and rows_deleted_notification > 0: flash("Notification cleared.", "warning")
            elif rows_deleted_notification == 0: flash("Action processed, but failed to clear notification.", "error")
            else: flash("Action processed.", "info")

     except Exception as e:
        flash("Error removing content.", "error")
        app.logger.exception(f"Removal error notification {notification_id}: {e}")
     return redirect(url_for('notifications'))

# --- Friends Routes (Modified for PostgreSQL) ---
@app.route('/friends')
def friends():
    if 'user_id' not in session: return redirect(url_for('login'))
    user_id = session['user_id']
    friend_requests = []
    friends_list = []
    search_results = request.args.get('search_results', None) # Check if redirected from search

    try:
        with get_connection() as (conn, cursor):
            # Get incoming friend requests
            cursor.execute("""
                SELECT fr.id, fr.sender_id, u.username as sender_username
                FROM "FriendRequests" fr JOIN "Users" u ON u.id = fr.sender_id
                WHERE fr.receiver_id = %s AND fr.status = 'pending'
            """, (user_id,))
            requests_raw = cursor.fetchall()
            friend_requests = [{'id': r[0], 'sender_id': r[1], 'sender_username': r[2]} for r in requests_raw]

            # Get accepted friends (more complex query needed)
            cursor.execute("""
                SELECT u.id, u.username
                FROM "Users" u
                WHERE u.id != %s AND EXISTS (
                    SELECT 1 FROM "FriendRequests" fr
                    WHERE fr.status = 'accepted' AND
                          ((fr.sender_id = %s AND fr.receiver_id = u.id) OR
                           (fr.sender_id = u.id AND fr.receiver_id = %s))
                )
            """, (user_id, user_id, user_id))
            friends_raw = cursor.fetchall()
            friends_list = [{'id': f[0], 'username': f[1]} for f in friends_raw]

    except Exception as e:
        app.logger.exception(f"Error loading friends page for user {user_id}: {e}")
        flash("Error loading friends data.", "error")

    # Handle displaying search results if they were passed via redirect/session (or query param)
    # For simplicity, we'll assume search results are passed directly if needed
    # This example doesn't re-fetch search results on GET to /friends

    return render_template('friends.html',
                           friend_requests=friend_requests,
                           friends=friends_list,
                           friends_count=len(friends_list),
                           # Pass search_results ONLY if redirected from search_users
                           search_results=search_results if search_results else [])


@app.route('/unfriend/<int:friend_id>') # Use POST for actions that change state
def unfriend(friend_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        with get_connection() as (conn, cursor):
            # Delete the accepted friendship record(s) - might be one row
            cursor.execute("""
                DELETE FROM "FriendRequests"
                WHERE status = 'accepted' AND
                      ((sender_id = %s AND receiver_id = %s) OR (sender_id = %s AND receiver_id = %s))
            """, (user_id, friend_id, friend_id, user_id))
            if cursor.rowcount > 0: flash("Friend removed successfully.", "success")
            else: flash("Friendship not found or already removed.", "warning")
    except Exception as e:
         app.logger.exception(f"Error unfriending {friend_id} for user {user_id}: {e}")
         flash("Error removing friend.", "error")
    return redirect(url_for('friends'))

@app.route('/accept_request/<int:request_id>') # Use POST
def accept_request(request_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    try:
        with get_connection() as (conn, cursor):
            cursor.execute("""
                UPDATE "FriendRequests" SET status = 'accepted'
                WHERE id = %s AND receiver_id = %s AND status = 'pending'
            """, (request_id, session['user_id']))
            if cursor.rowcount > 0: flash("Friend request accepted.", "success")
            else: flash("Request not found or already actioned.", "warning")
    except Exception as e:
        app.logger.exception(f"Error accepting request {request_id}: {e}")
        flash("Error accepting request.", "error")
    return redirect(url_for('friends'))

@app.route('/reject_request/<int:request_id>') # Use POST
def reject_request(request_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    try:
        with get_connection() as (conn, cursor):
            # Option 1: Delete the request
            cursor.execute('DELETE FROM "FriendRequests" WHERE id = %s AND receiver_id = %s',
                           (request_id, session['user_id']))
            # Option 2: Update status to 'rejected' (if you want to track it)
            # cursor.execute("UPDATE FriendRequests SET status = 'rejected' WHERE id = %s AND receiver_id = %s", (request_id, session['user_id']))
            if cursor.rowcount > 0: flash("Friend request rejected.", "info")
            else: flash("Request not found or already actioned.", "warning")
    except Exception as e:
         app.logger.exception(f"Error rejecting request {request_id}: {e}")
         flash("Error rejecting request.", "error")
    return redirect(url_for('friends'))

@app.route('/search_users', methods=['GET']) # Typically GET for search results display
def search_users():
    if 'user_id' not in session: return redirect(url_for('login'))
    query = request.args.get('query', '').strip()
    if not query:
        flash("Please enter a name to search.", "warning")
        return redirect(url_for('friends')) # Redirect back if query is empty

    user_id = session['user_id']
    search_results_list = []
    try:
        with get_connection() as (conn, cursor):
            # Search for users (case-insensitive using ILIKE)
            search_term = f'%{query}%'
            cursor.execute('SELECT id, username FROM "Users" WHERE username ILIKE %s AND id != %s',
                           (search_term, user_id))
            users_raw = cursor.fetchall()

            if not users_raw:
                 flash(f"No users found matching '{query}'.", "info")
                 return redirect(url_for('friends')) # Redirect if no results

            # Get IDs of found users to check friendship status efficiently
            found_user_ids = [u[0] for u in users_raw]

            # Check friendship/request status in one query
            friendship_statuses = {}
            cursor.execute("""
                SELECT sender_id, receiver_id, status
                FROM "FriendRequests"
                WHERE (sender_id = %s AND receiver_id = ANY(%s))
                   OR (receiver_id = %s AND sender_id = ANY(%s))
            """, (user_id, found_user_ids, user_id, found_user_ids))
            status_rows = cursor.fetchall()

            for row in status_rows:
                s_id, r_id, status = row
                other_user_id = r_id if s_id == user_id else s_id
                # Store status relative to the current user
                # 'sent': request sent by current user
                # 'received': request received by current user
                # 'accepted': they are friends
                if status == 'accepted':
                    friendship_statuses[other_user_id] = 'accepted'
                elif status == 'pending':
                    if s_id == user_id: friendship_statuses[other_user_id] = 'sent'
                    else: friendship_statuses[other_user_id] = 'received'
                # Ignore 'rejected' or other statuses for this display maybe

            # Build results list
            for u in users_raw:
                 other_id = u[0]
                 status = friendship_statuses.get(other_id, 'none') # 'none', 'accepted', 'sent', 'received'
                 search_results_list.append({
                     'id': other_id,
                     'username': u[1],
                     'status': status
                 })

    except Exception as e:
         app.logger.exception(f"Error searching users for query '{query}': {e}")
         flash("Error during user search.", "error")
         return redirect(url_for('friends'))

    # Render the *same* friends template, but pass the search results
    # The template needs logic to display search results when present
    return render_template('friends.html',
                           friend_requests=[], # Don't show requests when showing search results
                           friends=[],         # Don't show friend list when showing search results
                           friends_count=0,    # Or fetch separately if needed
                           search_results=search_results_list,
                           search_query=query) # Pass query back to display


@app.route('/send_request/<int:receiver_id>') # Use POST
def send_request(receiver_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    sender_id = session['user_id']

    if sender_id == receiver_id: # Prevent sending request to self
        flash("You cannot send a friend request to yourself.", "warning")
        return redirect(request.referrer or url_for('friends')) # Redirect back

    try:
        with get_connection() as (conn, cursor):
            # Check target user exists
            cursor.execute('SELECT id FROM "Users" WHERE id = %s', (receiver_id,))
            if not cursor.fetchone():
                 flash("User not found.", "error"); return redirect(url_for('friends'))

            # Check if already friends or request exists (pending or otherwise)
            cursor.execute("""
                SELECT status FROM "FriendRequests"
                WHERE (sender_id = %s AND receiver_id = %s) OR (sender_id = %s AND receiver_id = %s)
            """, (sender_id, receiver_id, receiver_id, sender_id))
            existing = cursor.fetchone()

            if existing:
                status = existing[0]
                if status == 'accepted': flash("You are already friends.", "info")
                elif status == 'pending':
                     # Check who sent it
                     cursor.execute("SELECT sender_id FROM \"FriendRequests\" WHERE (sender_id = %s AND receiver_id = %s) OR (sender_id = %s AND receiver_id = %s)",
                                    (sender_id, receiver_id, receiver_id, sender_id))
                     actual_sender = cursor.fetchone()[0]
                     if actual_sender == sender_id: flash("Friend request already sent.", "info")
                     else: flash("You already have a pending request from this user.", "info")
                elif status == 'rejected': flash("A previous request was rejected.", "info") # Or allow re-request?
                else: flash(f"Cannot send request due to existing status: {status}", "warning")
            else:
                # Insert new pending request
                cursor.execute("""
                    INSERT INTO "FriendRequests" (sender_id, receiver_id, status, created_at)
                    VALUES (%s, %s, 'pending', CURRENT_TIMESTAMP)
                """, (sender_id, receiver_id))
                flash("Friend request sent.", "success")
                # Consider sending a notification to the receiver
                # ... notification insert logic ...

    except Exception as e:
         app.logger.exception(f"Error sending friend request from {sender_id} to {receiver_id}: {e}")
         flash("Error sending friend request.", "error")

    # Redirect back to where the user was, or friends page as fallback
    return redirect(url_for('friends'))

# --- Profile & Settings Routes (Modified for PostgreSQL) ---
@app.route('/profile')
def profile():
    if 'user_id' not in session: return redirect(url_for('login'))
    user_id = session['user_id']
    user_data = None
    posts_count = 0
    friends_count = 0
    try:
        with get_connection() as (conn, cursor):
            cursor.execute('SELECT id, username, status FROM "Users" WHERE id = %s', (user_id,))
            user_raw = cursor.fetchone()
            if user_raw: user_data = {'id': user_raw[0], 'username': user_raw[1], 'status': user_raw[2]} # Convert to dict-like

            cursor.execute('SELECT COUNT(*) FROM "Posts" WHERE user_id = %s', (user_id,))
            posts_count = cursor.fetchone()[0]

            # Count accepted friends
            cursor.execute("""
                 SELECT COUNT(DISTINCT u.id) FROM "Users" u
                 WHERE u.id != %s AND EXISTS (
                     SELECT 1 FROM "FriendRequests" fr WHERE fr.status = 'accepted' AND
                           ((fr.sender_id = %s AND fr.receiver_id = u.id) OR (fr.sender_id = u.id AND fr.receiver_id = %s))
                 )
            """, (user_id, user_id, user_id))
            friends_count = cursor.fetchone()[0]

    except Exception as e:
        app.logger.exception(f"Error loading profile for user {user_id}: {e}")
        flash("Error loading profile.", "error")
        return redirect(url_for('dashboard')) # Redirect somewhere sensible on error

    if not user_data: return redirect(url_for('login')) # Should not happen if session is valid, but safety check

    return render_template('profile.html', user=user_data, posts_count=posts_count, friends_count=friends_count)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' not in session: return redirect(url_for('login'))
    user_id = session['user_id']

    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if new_password: # Only update if a new password was provided
            if not confirm_password:
                flash("Please confirm your new password.", "warning")
            elif new_password != confirm_password:
                flash("New passwords do not match.", "warning")
            elif len(new_password) < 6:
                 flash("Password must be at least 6 characters long.", "warning")
            else:
                try:
                    # !! IMPORTANT: Hash the new password !!
                    # from werkzeug.security import generate_password_hash
                    # hashed_password = generate_password_hash(new_password)
                    hashed_password = new_password # TEMPORARY - REPLACE WITH HASHING

                    with get_connection() as (conn, cursor):
                        cursor.execute('UPDATE "Users" SET password = %s WHERE id = %s', (hashed_password, user_id))
                    flash("Password updated successfully.", "success")
                    # Stay on settings page or redirect? Redirect often preferred after success.
                    return redirect(url_for('settings'))
                except Exception as e:
                    app.logger.exception(f"Error updating password for user {user_id}: {e}")
                    flash("Error updating password.", "error")
        else:
             flash("No new password entered.", "info") # Feedback if fields were empty

    # GET request just shows the page
    return render_template('settings.html')

@app.route('/view_profile/<int:user_id>')
def view_profile(user_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    current_user_id = session['user_id']
    
    with get_connection() as (conn, cursor):
        # Check if the user exists
        
        # Get target user info
        cursor.execute('SELECT id, username, status FROM "Users" WHERE id = %s', (user_id,))
        user_tuple = cursor.fetchone()
        
        if not user_tuple:
            flash("User not found.", "error")
            return redirect(url_for('friends'))
        
        # Convert tuple to dictionary
        profile_user = {
            'id': user_tuple[0],
            'username': user_tuple[1],
            'status': user_tuple[2]
        }
        
        # Check if they are friends
        cursor.execute('SELECT status FROM "FriendRequests" WHERE ((sender_id = %s AND receiver_id = %s) OR (sender_id = %s AND receiver_id = %s))', (current_user_id, user_id, user_id, current_user_id))
        friendship = cursor.fetchone()
        
        friendship_status = friendship[0] if friendship else None
        
        # Get their recent posts (only if friends or viewing own profile)
        posts = []
        if user_id == current_user_id or (friendship and friendship_status == 'accepted'):
            cursor.execute('SELECT id, content, media_url, media_type, created_at FROM "Posts" WHERE user_id = %s ORDER BY created_at DESC', (user_id,))
            
            posts = [
                {
                    'id': row[0],  # Access by index if cursor returns tuples
                    'content': row[1],
                    'media_url': row[2],
                    'media_type': row[3],
                    'created_at': row[4]
                }
                for row in cursor.fetchall()
            ]
    
    return render_template('view_profile.html', 
                          profile_user=profile_user, 
                          posts=posts,
                          friendship_status=friendship_status)


# --- Chatbot Route (using CrewAI logic) ---
@app.route('/chatbot', methods=['POST'])
def chatbot(): # Keep specific name
    try:
        data = request.json
        if not data or 'message' not in data:
             app.logger.warning("Chatbot request received without data or message key.")
             return jsonify({'response': 'Invalid request format.'}), 400
        user_message = data.get('message', '').strip()
        if not user_message:
             app.logger.info("Chatbot request received with empty message.")
             return jsonify({'response': 'Please type a message.'}), 400

        if needs_search(user_message):
            response_text = handle_search_query(user_message)
        else:
            response_text = handle_conversation(user_message)
        return jsonify({'response': response_text})
    except Exception as e:
        app.logger.error(f"Unhandled error in /chatbot route: {e}", exc_info=True)
        return jsonify({'response': 'Sorry, a server error occurred.'}), 500


# --- Main execution ---
if __name__ == '__main__':
    # Consider setting debug based on an environment variable for production
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't'])
