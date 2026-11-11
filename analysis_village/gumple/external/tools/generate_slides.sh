#!/bin/bash
#
# Bash Script: generate_slides.sh
# Description: Reads files from a specified directory and generates a
#              LaTeX Beamer presentation, creating one slide per file.
#              This version focuses on generating the .tex source file only.
#
# Usage: ./generate_slides.sh <directory_to_scan>
#
# NOTE: To produce a PDF, you must manually run 'pdflatex presentation.tex' (twice)
# after running this script.

# --- Configuration ---
TARGET_DIR="$1"
OUTPUT_TEX="presentation.tex"
LATEX_TITLE="Directory Contents Overview"
LATEX_AUTHOR="Automated Script"
LATEX_THEME="Madrid" # You can change this theme (e.g., 'Darmstadt', 'AnnArbor')

# Function to display error and exit
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# --- Validation ---

# Check if a directory path was provided
if [ -z "$TARGET_DIR" ]; then
    error_exit "Please provide a directory path as the first argument."
fi

# Check if the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    error_exit "Directory '$TARGET_DIR' not found."
fi

# Ensure the output file is writable
if [ -f "$OUTPUT_TEX" ] && [ ! -w "$OUTPUT_TEX" ]; then
    error_exit "Output file '$OUTPUT_TEX' exists but is not writable."
fi

echo "--- Starting Slide Generation ---"
echo "Target Directory: $TARGET_DIR"

# --- 1. Generate the LaTeX (.tex) file using a here-document ---
generate_latex() {
    # Start the LaTeX content generation, writing to $OUTPUT_TEX
    cat << EOL > "$OUTPUT_TEX"
\documentclass{beamer}

% Preamble
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx} % ADDED: Required for \includegraphics
\usepackage{listings}
\usepackage{grffile}  % RECOMMENDED: Better handling of complex filenames in \includegraphics

% Set the visual theme
\usetheme{$LATEX_THEME}
\usecolortheme{whale} % Example color theme
\setbeamertemplate{navigation symbols}{} % Hide navigation buttons

\title{$LATEX_TITLE}
\author{$LATEX_AUTHOR}
\date{\today}

\begin{document}

% ----------------- Title Slide -----------------
\begin{frame}
  \titlepage
\end{frame}

% ----------------- Overview Slide -----------------
\begin{frame}{Files Found in \texttt{$TARGET_DIR}}
  \frametitle{File Inventory}
  The following files were detected and added to this presentation:
  \begin{itemize}
EOL

    # Loop through the files in the directory (excluding hidden files)
    find "$TARGET_DIR" -maxdepth 1 -type f ! -name ".*" -print0 | while IFS= read -r -d $'\0' FILE; do
        # Extract just the filename from the full path
        FILENAME=$(basename "$FILE")
        
        # ESCAPING: Replace underscores with \_ for safe display in LaTeX text environments
        DISPLAY_FILENAME=$(echo "$FILENAME" | sed 's/_/\\_/g')
        
        # Add the filename to the overview list
        echo "    \item \texttt{$DISPLAY_FILENAME}" >> "$OUTPUT_TEX"
    done

    # Close the itemize environment for the overview slide
    cat << EOL_OVERVIEW_END >> "$OUTPUT_TEX"
  \end{itemize}
\end{frame}

% ----------------- Individual Slides per File -----------------
EOL_OVERVIEW_END

    # Loop through the files again to create individual slides
    find "$TARGET_DIR" -maxdepth 1 -type f ! -name ".*" -print0 | while IFS= read -r -d $'\0' FILE; do
        FILENAME=$(basename "$FILE")
        
        # ESCAPING: Replace underscores with \_ for safe display in LaTeX text environments
        DISPLAY_FILENAME=$(echo "$FILENAME" | sed 's/_/\\_/g')
        DISPLAY_FILEPATH=$(echo "$FILE" | sed 's/_/\\_/g')

        # Check for PDF extension (case insensitive)
        if [[ "$FILENAME" =~ \.[pP][dD][fF]$ ]]; then
            # Use \includegraphics for PDF files
            SLIDE_CONTENT="
  \begin{center}
    % \includegraphics needs the UNESCAPED file path
    \includegraphics[width=\\textwidth, height=0.8\\textheight, keepaspectratio]{$FILE}
  \end{center}
  \vspace{0.2cm}
  \begin{alertblock}{File Type}
    PDF content embedded using \texttt{\string\includegraphics}.
  \end{alertblock}"
        else
            # Default slide for other file types
            SLIDE_CONTENT="
  This slide is dedicated to the file:
  \begin{itemize}
    \item \textbf{Name}: \texttt{$DISPLAY_FILENAME}
    \item \textbf{Full Path}: \texttt{$DISPLAY_FILEPATH}
  \end{itemize}

  \begin{block}{Content Summary}
    This is a generic file type. Content embedding is not supported for this format.
  \end{block}"
        fi
        
        # Generate a slide for each file
        cat << EOL_FRAME >> "$OUTPUT_TEX"
\begin{frame}
  \frametitle{File: \texttt{$DISPLAY_FILENAME}}
  \centering
  \vspace{0.5cm}
  
$SLIDE_CONTENT

\end{frame}
EOL_FRAME
    done

    # --- Close the document ---
    cat << EOL_END >> "$OUTPUT_TEX"
\end{document}
EOL_END

    echo "Successfully generated LaTeX source file: $OUTPUT_TEX"
    echo "To produce the PDF, run: pdflatex presentation.tex"
}

# --- Main Execution ---
generate_latex

echo "--- Finished ---"
