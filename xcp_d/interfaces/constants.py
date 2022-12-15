"""Constants to be used by interfaces."""
IMAGE_INFO = {
    "t1w_brainplot": {"pattern": "*desc-brainsprite_T1w.html", "title": "T1w BrainSprite"},
    "t2w_brainplot": {"pattern": "*desc-brainsprite_T1w.html", "title": "T2w BrainSprite"},
    "task_pre_reg_gray": {"pattern": "*%s*desc-precarpetplot_bold.svg", "title": "Pre-Regression"},
    "task_post_reg_gray": {
        "pattern": "*%s*desc-postcarpetplot_bold.svg",
        "title": "Post-Regression",
    },
    "bold_t1w_reg": {"pattern": "*%s*desc-bbregister*bold.svg", "title": "Bold T1w registration"},
    "ref": {"pattern": "*%s*desc-boldref*bold.svg", "title": "Reference"},
}

# HTML constants:
HTML_START = """
<!DOCTYPE html>
<html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
<link rel="stylesheet"
href="https://use.fontawesome.com/releases/v5.7.0/css/all.css"
    integrity="sha384-lZN37f5QGtY3VHgisS14W3ExzMWZxybE1SJSEsQp9S+oqd12jhcu+A56Ebc1zFSJ"
    crossorigin="anonymous">
<style type="text/css">
    header, footer, section, article, nav, aside { display: block; }
    h1, h2, h3, body, button, p, w3-btn { font-family: Verdana, Helvetica,
    Arial, Bookman, sans-serif; }
    h1 { text-align: center; font-size: 2.5em; }
    h2{ text-align: center; font-size: 2.00em; }
    h3{ text-align: left; font-size: 2.5em; }
    p{ font-size: 1.0em; }
    body{ font-size: 1.0em; }
    img{ width: 100%; padding: 2px; }
    .label1, .label2, .label3, .label4{ font-family: Verdana, Helvetica,
     Arial, Bookman, sans-serif }
    .label1 { font-size: 1.25em; text-align: center; }
    .label2 { font-size: 1.25em; text-align: right; }
    .label3 { font-size: 1.25em; text-align: left; }
    .label4 { font-size: 1.00em; text-align: center; }
    .grid-container { grid-gap: 2px; padding: 2px; }
    .T1pngs, .T2pngs, .Registrations, .Images { display: none; }
    .modal { vertical-align: top; margin-top:0; border-top-style:none;
     padding-top:0; top:0; height: 100%; width: auto; }
    .Images{ height: 100%; width: auto; }
</style>
<body>
"""

# Make the html 'title' (what will be seen on the tab, etc.),
# as well as the page header.
# Needs the following values:
#    subject, session.
TITLE = """
<title>Executive Summary: {subject} {session}</title>
<header> <h1>{subject}{sep}{session}</h1> </header>
"""

HTML_END = """
</body>
</html>
"""

# Make a section for T1 or T2. This will show the brainsprite
# viewer, with a label above it to identify T1 v T2, and a
# button that will open a modal window with a slider for the
# T1 or T2 pngs.
# Needs the following values:
#    tx, brainsprite_label, modal_button, brainsprite_viewer.
TX_SECTION_START = """
<section id="{txx}">
    <div class="w3-container">
        <div class="w3-row-padding">
            <div class="w3-center"><h2>Anatomical Data</h2></div>
        <div>

        </div>
</section>
        """

T1X_SECTION = """
<section id="{tx1}">
    <div class="w3-col l1 label2"> {tx1} </div>
    <div  {t1wbrainplot} </div>
</section>
"""

T2X_SECTION = """
<section id="{tx2}">
    <div class="w3-col l1 label2"> {tx2} </div>
    <div {t2wbrainplot} </div>
</section>

"""

TX_SECTION_END = """

"""
# Layout images in different formats - row, half, quarter.
# Needs the following values:
#    row_label, row_img, row_modal, row_idx
# Modal and idx are the slider and the index into the slider (or other modal
# container) to which the image was added. If the user clicks on the image, html
# will open the modal container to the index.
LAYOUT_ROW = """
        <div  class="w3-row-padding">
            <div class="w3-col l1 label2">{row_label}</div>
            <div class="w3-col l11"><img src="{row_img}"
            onclick="open_{row_modal}_to_index({row_idx})"></div>
        </div>
        """

LAYOUT_HALF_ROW = """
        <div  class="w3-row-padding">
            <div class="w3-col l2 label2">{row_label}</div>
            <div class="w3-col l9"><img src="{row_img}"
            onclick="open_{row_modal}_to_index({row_idx})"></div>
        </div>
        """

LAYOUT_QUARTER_ROW = """
            <div class="w3-quarter">
                <div class="w3-row w3-center label1">{row_label}</div>
                <div class="w3-row"><img src="{row_img}" onclick="open
                _{row_modal}_to_index({row_idx})"></div>
            </div>
            """

# Placeholders for rows of images.
# Needs the following value:
#    row_label
PLACEHOLDER_ROW = """
        <div class="w3-row-padding">
            <div class="w3-col l1 label2">{row_label}</div>
            <div class="w3-col l11">
                <div class="w3-container w3-pale-red label3">Image Not Available</div>
                <div class="w3-container w3-pale-red label3"><br></div>
            </div>
            <div class="w3-container"><br></div>
        </div>
        """

PLACEHOLDER_HALF_ROW = """
                 <div class="w3-row">
                     <div class="w3-col l2 label2">{row_label}</div>
                     <div class="w3-col l9">
                         <div class="w3-container w3-pale-red label3">Image Not Available</div>
                         <div class="w3-container w3-pale-red label3"><br></div>
                     </div>
                     <div class="w3-container"><br></div>
                 </div>
                 """

PLACEHOLDER_QUARTER_ROW = """
            <div class="w3-quarter">
                <div class="w3-row w3-center label1">{row_label}</div>
                <div class="w3-container w3-pale-red label3">Image Not Available</div>
                <div class="w3-container w3-pale-red label3"><br></div>
            </div>
            """

# Make a section for anatomical data (cortical and
# subcorticals aligned to atlases, combined gray ordinates).
# Start with the column headings.
# No values needed.
ANAT_SECTION_START = """
<section id="Anat">
    <div class="w3-container">
        <div class="w3-row-padding">
            <div class="w3-center"><h2>Anatomical Data</h2></div>
        </div>
            """

# Layout the row of gray-ordinates images.
GRAY_ROW_START = """
    <div class="w3-row-padding">
        <div class="w3-col l1 label2"><br>Combined Resting State Data</div>
"""
GRAY_ROW_END = """
        </div>
    </div>
"""

# End the atlas section by closing up the divisions and the section.
# No values needed.
ANAT_SECTION_END = """
    </div>
</section>
"""

# Start the tasks section and  put in the column headings for the task-specific data.
TASKS_SECTION_START = """
<section id="Tasks">
    <div class="w3-container">
        <div class="w3-row-padding">
            <div class="w3-center"><h2>Functional Data</h2></div>
        <div>
        """

# Add the task name for the next few rows.
TASK_LABEL_ROW = """
        <div  class="w3-row">
            <div class="w3-left label2">task-{task_name} run-{task_num}:</div>
        <div  class="w3-row">
        """

BOLD_GRAY_START = """
        <div class="w3-row-padding">
            <div class="w3-half">
        """

BOLD_GRAY_SPLIT = """
            </div>
            """

BOLD_GRAY_END = """
        </div>
        """

# Layout the row of bold, reference and gray-ordinates images for the task.
# Needs the following values and corresponding indices:
#    modal_id, bold, ref, task_pre_reg_gray, task_post_reg_gray
# BOLD_GRAY_ROW2="""
# <div class="w3-quarter">
# <div class="w3-row w3-center label1">Pre-Regression</div>
# <div class="w3-row"><img src="{task_pre_reg_gray}"
# onclick="open_{modal_id}_to_index({task_pre_reg_gray_idx})"></div>
# </div>
# <div class="w3-quarter">
# <div class="w3-row w3-center label1">Post-Regression</div>
# <div class="w3-row"><img src="{task_post_reg_gray}"
# onclick="open_{modal_id}_to_index({task_post_reg_gray_idx})"></div>
# </div>
# """

# Close up the divisions and section.
TASKS_SECTION_END = """
    </div>
</section>
"""

# MODAL/SLIDER STUFF

# Begin a modal container.
# Needs the following values:
#    modal_id
MODAL_START = """
    <div id="{modal_id}" class="w3-modal">
        <div class="w3-modal-content">
            <div class="w3-content w3-display-container">
            """

# Make a button to display a modal container.
# Needs the following values:
#    modal_id, btn_label
DISPLAY_MODAL_BUTTON = """
            <button class="w3-btn w3-teal"
            onclick="open_{modal_id}_to_index(1)">{btn_label}</button>
            """

# Add the containers' buttons at the end, so that they don't
# get covered by the images. Every slider needs a left and
# right button.
# Needs the following values:
#    image_class
SLIDER_END = """
                <button class="w3-button w3-black w3-display-bottomleft w3-xxlarge"
                    onclick="change_{image_class}(-1)"><i class="fas fa-angle-left"></i></button>
                <button class="w3-button w3-black w3-display-bottomright w3-xxlarge"
                    onclick="change_{image_class}(1)"><i class="fas fa-angle-right"></i></button>
                    """
# The modal window needs a close button; then, close up the
# elements.
# Needs the following values:
#    modal_id
MODAL_END = """
                <button class="w3-btn w3-red w3-display-topright w3-large"
                    onclick="document.getElementById('{modal_id}').style.display='none'">
                    <i class="fa fa-close"></i></button>
            </div>
        </div>
    </div>
"""

# Add an image in a container with its filename displayed in the
# upper left corner. The filename is 'w3-black' so that the text
# will be white and show up against the fMRI image without being
# too obtrusive.
# We assign a specific class so that scripts can find the images
# by calling getElementsByClassName().
# Needs the following values:
#    image_class, image_file, display_name.
IMAGE_WITH_CLASS = """
                <div class="w3-display-container {image_class}">
                    <object type="image/svg+xml" data={image_file}>
                    <img src={image_file}/> </object>
                    <div class="w3-display-topleft w3-black"><p>{display_name}</p></div>
                </div>
                """

# The modal scripts that will show the chosen image.
# Needs the following values:
#    modal_id, image_class.
MODAL_SCRIPTS = """
<script>
    var %(image_class)sIdx = 1;
    show_%(image_class)s(%(image_class)sIdx)

    function show_%(image_class)s(n) {
        var i;
        var x = document.getElementsByClassName("%(image_class)s");
        if (n > x.length) { %(image_class)sIdx = 1 }
        if (n < 1) { %(image_class)sIdx = x.length }
        for (i = 0; i < x.length; i++) {
            x[i].style.display = "none";
        }
        x[%(image_class)sIdx-1].style.display = "block";
    }

    function open_%(modal_id)s_to_index(idx) {
        show_%(image_class)s(%(image_class)sIdx = idx)
        document.getElementById("%(modal_id)s").style.display='block'
    }
</script>
"""

# The slider scripts that will show the next or previous image
# in a given class.
# Needs the following values:
#    modal_id, image_class.
SLIDER_SCRIPTS = """
<script>
    var %(image_class)sIdx = 1;
    show_%(image_class)s(%(image_class)sIdx)

    function change_%(image_class)s(n) {
        show_%(image_class)s(%(image_class)sIdx += n)
    }

    function show_%(image_class)s(n) {
        var i;
        var x = document.getElementsByClassName("%(image_class)s");
        if (n > x.length) { %(image_class)sIdx = 1 }
        if (n < 1) { %(image_class)sIdx = x.length }
        for (i = 0; i < x.length; i++) {
            x[i].style.display = "none";
        }
        x[%(image_class)sIdx-1].style.display = "block";
    }

    function open_%(modal_id)s_to_index(idx) {
        show_%(image_class)s(%(image_class)sIdx = idx)
        document.getElementById("%(modal_id)s").style.display='block'
    }
</script>
"""

#
