namespace StableDiffusionSharpGUI
{
    partial class MainForm
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.label3 = new System.Windows.Forms.Label();
            this.TextBox_ModelPath = new System.Windows.Forms.TextBox();
            this.Button_ScanModelPath = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.TextBox_VaePath = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.TextBox_LoraModelDir = new System.Windows.Forms.TextBox();
            this.Button_ScanVaePath = new System.Windows.Forms.Button();
            this.Button_ScanLoraPath = new System.Windows.Forms.Button();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.CheckBox_VaeTiling = new System.Windows.Forms.CheckBox();
            this.CheckBox_CpuVae = new System.Windows.Forms.CheckBox();
            this.Button_LoadModel = new System.Windows.Forms.Button();
            this.ProgressBar_Progress = new System.Windows.Forms.ProgressBar();
            this.label11 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.label13 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.TabPage_Upscale = new System.Windows.Forms.TabPage();
            this.Button_Upscale = new System.Windows.Forms.Button();
            this.Button_ScanUpscaleModelPath = new System.Windows.Forms.Button();
            this.TextBox_UpscaleModelPath = new System.Windows.Forms.TextBox();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.PictureBox_UpscaleOutput = new System.Windows.Forms.PictureBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.PictureBox_UpscaleInput = new System.Windows.Forms.PictureBox();
            this.label18 = new System.Windows.Forms.Label();
            this.TabPage_CreateImage = new System.Windows.Forms.TabPage();
            this.PictureBox_OutputImage = new System.Windows.Forms.PictureBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.NumericUpDown_ReDrawStrength = new System.Windows.Forms.NumericUpDown();
            this.label17 = new System.Windows.Forms.Label();
            this.NumericUpDown_ClipSkip = new System.Windows.Forms.NumericUpDown();
            this.label16 = new System.Windows.Forms.Label();
            this.PictureBox_InputImage = new System.Windows.Forms.PictureBox();
            this.Button_ImageToImage = new System.Windows.Forms.Button();
            this.Button_RandomSeed = new System.Windows.Forms.Button();
            this.NumericUpDown_Seed = new System.Windows.Forms.NumericUpDown();
            this.label15 = new System.Windows.Forms.Label();
            this.NumericUpDown_SampleSteps = new System.Windows.Forms.NumericUpDown();
            this.label10 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.NumericUpDown_Height = new System.Windows.Forms.NumericUpDown();
            this.NumericUpDown_Width = new System.Windows.Forms.NumericUpDown();
            this.NumericUpDown_CFG = new System.Windows.Forms.NumericUpDown();
            this.label7 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.ComboBox_SampleMethod = new System.Windows.Forms.ComboBox();
            this.label1 = new System.Windows.Forms.Label();
            this.Button_TextToImage = new System.Windows.Forms.Button();
            this.TextBox_Prompt = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.TextBox_NegativePrompt = new System.Windows.Forms.TextBox();
            this.TabControl = new System.Windows.Forms.TabControl();
            this.label19 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.TabPage_Upscale.SuspendLayout();
            this.groupBox5.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.PictureBox_UpscaleOutput)).BeginInit();
            this.groupBox4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.PictureBox_UpscaleInput)).BeginInit();
            this.TabPage_CreateImage.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.PictureBox_OutputImage)).BeginInit();
            this.groupBox2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_ReDrawStrength)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_ClipSkip)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.PictureBox_InputImage)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_Seed)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_SampleSteps)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_Height)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_Width)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_CFG)).BeginInit();
            this.TabControl.SuspendLayout();
            this.SuspendLayout();
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(9, 40);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(65, 12);
            this.label3.TabIndex = 7;
            this.label3.Text = "Model Path";
            // 
            // TextBox_ModelPath
            // 
            this.TextBox_ModelPath.Location = new System.Drawing.Point(86, 31);
            this.TextBox_ModelPath.Name = "TextBox_ModelPath";
            this.TextBox_ModelPath.Size = new System.Drawing.Size(417, 21);
            this.TextBox_ModelPath.TabIndex = 8;
            // 
            // Button_ScanModelPath
            // 
            this.Button_ScanModelPath.Location = new System.Drawing.Point(516, 29);
            this.Button_ScanModelPath.Name = "Button_ScanModelPath";
            this.Button_ScanModelPath.Size = new System.Drawing.Size(75, 23);
            this.Button_ScanModelPath.TabIndex = 9;
            this.Button_ScanModelPath.Text = "Scan";
            this.Button_ScanModelPath.UseVisualStyleBackColor = true;
            this.Button_ScanModelPath.Click += new System.EventHandler(this.Button_ScanModelPath_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(21, 80);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(53, 12);
            this.label4.TabIndex = 10;
            this.label4.Text = "VAE Path";
            // 
            // TextBox_VaePath
            // 
            this.TextBox_VaePath.Location = new System.Drawing.Point(86, 74);
            this.TextBox_VaePath.Name = "TextBox_VaePath";
            this.TextBox_VaePath.Size = new System.Drawing.Size(417, 21);
            this.TextBox_VaePath.TabIndex = 11;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(15, 117);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(59, 12);
            this.label5.TabIndex = 12;
            this.label5.Text = "Lora Path";
            // 
            // TextBox_LoraModelDir
            // 
            this.TextBox_LoraModelDir.Location = new System.Drawing.Point(86, 114);
            this.TextBox_LoraModelDir.Name = "TextBox_LoraModelDir";
            this.TextBox_LoraModelDir.Size = new System.Drawing.Size(417, 21);
            this.TextBox_LoraModelDir.TabIndex = 13;
            // 
            // Button_ScanVaePath
            // 
            this.Button_ScanVaePath.Location = new System.Drawing.Point(516, 72);
            this.Button_ScanVaePath.Name = "Button_ScanVaePath";
            this.Button_ScanVaePath.Size = new System.Drawing.Size(75, 23);
            this.Button_ScanVaePath.TabIndex = 14;
            this.Button_ScanVaePath.Text = "Scan";
            this.Button_ScanVaePath.UseVisualStyleBackColor = true;
            this.Button_ScanVaePath.Click += new System.EventHandler(this.Button_ScanVaePath_Click);
            // 
            // Button_ScanLoraPath
            // 
            this.Button_ScanLoraPath.Location = new System.Drawing.Point(516, 117);
            this.Button_ScanLoraPath.Name = "Button_ScanLoraPath";
            this.Button_ScanLoraPath.Size = new System.Drawing.Size(75, 23);
            this.Button_ScanLoraPath.TabIndex = 15;
            this.Button_ScanLoraPath.Text = "Scan";
            this.Button_ScanLoraPath.UseVisualStyleBackColor = true;
            this.Button_ScanLoraPath.Click += new System.EventHandler(this.Button_ScanLoraPath_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.CheckBox_VaeTiling);
            this.groupBox1.Controls.Add(this.CheckBox_CpuVae);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.Button_LoadModel);
            this.groupBox1.Controls.Add(this.Button_ScanLoraPath);
            this.groupBox1.Controls.Add(this.TextBox_ModelPath);
            this.groupBox1.Controls.Add(this.Button_ScanVaePath);
            this.groupBox1.Controls.Add(this.Button_ScanModelPath);
            this.groupBox1.Controls.Add(this.TextBox_LoraModelDir);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.TextBox_VaePath);
            this.groupBox1.Location = new System.Drawing.Point(12, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(754, 147);
            this.groupBox1.TabIndex = 16;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Base";
            // 
            // CheckBox_VaeTiling
            // 
            this.CheckBox_VaeTiling.AutoSize = true;
            this.CheckBox_VaeTiling.Checked = true;
            this.CheckBox_VaeTiling.CheckState = System.Windows.Forms.CheckState.Checked;
            this.CheckBox_VaeTiling.Location = new System.Drawing.Point(612, 76);
            this.CheckBox_VaeTiling.Name = "CheckBox_VaeTiling";
            this.CheckBox_VaeTiling.Size = new System.Drawing.Size(84, 16);
            this.CheckBox_VaeTiling.TabIndex = 19;
            this.CheckBox_VaeTiling.Text = "Vae Tiling";
            this.CheckBox_VaeTiling.UseVisualStyleBackColor = true;
            // 
            // CheckBox_CpuVae
            // 
            this.CheckBox_CpuVae.AutoSize = true;
            this.CheckBox_CpuVae.Location = new System.Drawing.Point(612, 33);
            this.CheckBox_CpuVae.Name = "CheckBox_CpuVae";
            this.CheckBox_CpuVae.Size = new System.Drawing.Size(66, 16);
            this.CheckBox_CpuVae.TabIndex = 18;
            this.CheckBox_CpuVae.Text = "CPU VAE";
            this.CheckBox_CpuVae.UseVisualStyleBackColor = true;
            // 
            // Button_LoadModel
            // 
            this.Button_LoadModel.Location = new System.Drawing.Point(612, 114);
            this.Button_LoadModel.Name = "Button_LoadModel";
            this.Button_LoadModel.Size = new System.Drawing.Size(78, 23);
            this.Button_LoadModel.TabIndex = 17;
            this.Button_LoadModel.Text = "Load Model";
            this.Button_LoadModel.UseVisualStyleBackColor = true;
            this.Button_LoadModel.Click += new System.EventHandler(this.Button_LoadModel_Click);
            // 
            // ProgressBar_Progress
            // 
            this.ProgressBar_Progress.Location = new System.Drawing.Point(316, 20);
            this.ProgressBar_Progress.Name = "ProgressBar_Progress";
            this.ProgressBar_Progress.Size = new System.Drawing.Size(438, 23);
            this.ProgressBar_Progress.Step = 100;
            this.ProgressBar_Progress.TabIndex = 20;
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(127, 31);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(23, 12);
            this.label11.TabIndex = 21;
            this.label11.Text = "1/1";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(15, 31);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(53, 12);
            this.label12.TabIndex = 22;
            this.label12.Text = "Progress";
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(202, 31);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(35, 12);
            this.label13.TabIndex = 23;
            this.label13.Text = "Speed";
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(254, 31);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(41, 12);
            this.label14.TabIndex = 24;
            this.label14.Text = "1 it/s";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.label12);
            this.groupBox3.Controls.Add(this.label11);
            this.groupBox3.Controls.Add(this.ProgressBar_Progress);
            this.groupBox3.Controls.Add(this.label14);
            this.groupBox3.Controls.Add(this.label13);
            this.groupBox3.Location = new System.Drawing.Point(12, 610);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(760, 51);
            this.groupBox3.TabIndex = 26;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Progress";
            // 
            // TabPage_Upscale
            // 
            this.TabPage_Upscale.Controls.Add(this.Button_Upscale);
            this.TabPage_Upscale.Controls.Add(this.Button_ScanUpscaleModelPath);
            this.TabPage_Upscale.Controls.Add(this.TextBox_UpscaleModelPath);
            this.TabPage_Upscale.Controls.Add(this.groupBox5);
            this.TabPage_Upscale.Controls.Add(this.groupBox4);
            this.TabPage_Upscale.Controls.Add(this.label18);
            this.TabPage_Upscale.Location = new System.Drawing.Point(4, 22);
            this.TabPage_Upscale.Name = "TabPage_Upscale";
            this.TabPage_Upscale.Padding = new System.Windows.Forms.Padding(3);
            this.TabPage_Upscale.Size = new System.Drawing.Size(752, 413);
            this.TabPage_Upscale.TabIndex = 2;
            this.TabPage_Upscale.Text = "Upscale";
            this.TabPage_Upscale.UseVisualStyleBackColor = true;
            // 
            // Button_Upscale
            // 
            this.Button_Upscale.Location = new System.Drawing.Point(643, 31);
            this.Button_Upscale.Name = "Button_Upscale";
            this.Button_Upscale.Size = new System.Drawing.Size(75, 23);
            this.Button_Upscale.TabIndex = 7;
            this.Button_Upscale.Text = "Upscale";
            this.Button_Upscale.UseVisualStyleBackColor = true;
            this.Button_Upscale.Click += new System.EventHandler(this.Button_Upscale_Click);
            // 
            // Button_ScanUpscaleModelPath
            // 
            this.Button_ScanUpscaleModelPath.Location = new System.Drawing.Point(532, 31);
            this.Button_ScanUpscaleModelPath.Name = "Button_ScanUpscaleModelPath";
            this.Button_ScanUpscaleModelPath.Size = new System.Drawing.Size(75, 23);
            this.Button_ScanUpscaleModelPath.TabIndex = 6;
            this.Button_ScanUpscaleModelPath.Text = "Scan";
            this.Button_ScanUpscaleModelPath.UseVisualStyleBackColor = true;
            this.Button_ScanUpscaleModelPath.Click += new System.EventHandler(this.Button_ScanUpscaleModelPath_Click);
            // 
            // TextBox_UpscaleModelPath
            // 
            this.TextBox_UpscaleModelPath.Location = new System.Drawing.Point(98, 33);
            this.TextBox_UpscaleModelPath.Name = "TextBox_UpscaleModelPath";
            this.TextBox_UpscaleModelPath.Size = new System.Drawing.Size(391, 21);
            this.TextBox_UpscaleModelPath.TabIndex = 5;
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.PictureBox_UpscaleOutput);
            this.groupBox5.Location = new System.Drawing.Point(388, 67);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(330, 330);
            this.groupBox5.TabIndex = 4;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "Output Image";
            // 
            // PictureBox_UpscaleOutput
            // 
            this.PictureBox_UpscaleOutput.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.PictureBox_UpscaleOutput.Location = new System.Drawing.Point(13, 20);
            this.PictureBox_UpscaleOutput.Name = "PictureBox_UpscaleOutput";
            this.PictureBox_UpscaleOutput.Size = new System.Drawing.Size(300, 300);
            this.PictureBox_UpscaleOutput.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.PictureBox_UpscaleOutput.TabIndex = 1;
            this.PictureBox_UpscaleOutput.TabStop = false;
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.PictureBox_UpscaleInput);
            this.groupBox4.Location = new System.Drawing.Point(25, 67);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(330, 330);
            this.groupBox4.TabIndex = 3;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Input Image (Please drag and drop an image)";
            // 
            // PictureBox_UpscaleInput
            // 
            this.PictureBox_UpscaleInput.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.PictureBox_UpscaleInput.InitialImage = null;
            this.PictureBox_UpscaleInput.Location = new System.Drawing.Point(12, 20);
            this.PictureBox_UpscaleInput.Name = "PictureBox_UpscaleInput";
            this.PictureBox_UpscaleInput.Size = new System.Drawing.Size(300, 300);
            this.PictureBox_UpscaleInput.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.PictureBox_UpscaleInput.TabIndex = 0;
            this.PictureBox_UpscaleInput.TabStop = false;
            this.PictureBox_UpscaleInput.Click += new System.EventHandler(this.PictureBox_UpscaleInput_Click);
            this.PictureBox_UpscaleInput.DragDrop += new System.Windows.Forms.DragEventHandler(this.PictureBox_UpscaleInput_DragDrop);
            this.PictureBox_UpscaleInput.DragEnter += new System.Windows.Forms.DragEventHandler(this.PictureBox_UpscaleInput_DragEnter);
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(29, 36);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(65, 12);
            this.label18.TabIndex = 2;
            this.label18.Text = "Load Model";
            // 
            // TabPage_CreateImage
            // 
            this.TabPage_CreateImage.Controls.Add(this.PictureBox_OutputImage);
            this.TabPage_CreateImage.Controls.Add(this.groupBox2);
            this.TabPage_CreateImage.Location = new System.Drawing.Point(4, 22);
            this.TabPage_CreateImage.Name = "TabPage_CreateImage";
            this.TabPage_CreateImage.Padding = new System.Windows.Forms.Padding(3);
            this.TabPage_CreateImage.Size = new System.Drawing.Size(752, 413);
            this.TabPage_CreateImage.TabIndex = 0;
            this.TabPage_CreateImage.Text = "Create Images";
            this.TabPage_CreateImage.UseVisualStyleBackColor = true;
            // 
            // PictureBox_OutputImage
            // 
            this.PictureBox_OutputImage.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.PictureBox_OutputImage.Location = new System.Drawing.Point(381, 6);
            this.PictureBox_OutputImage.Name = "PictureBox_OutputImage";
            this.PictureBox_OutputImage.Size = new System.Drawing.Size(365, 399);
            this.PictureBox_OutputImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.PictureBox_OutputImage.TabIndex = 0;
            this.PictureBox_OutputImage.TabStop = false;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.label19);
            this.groupBox2.Controls.Add(this.NumericUpDown_ReDrawStrength);
            this.groupBox2.Controls.Add(this.label17);
            this.groupBox2.Controls.Add(this.NumericUpDown_ClipSkip);
            this.groupBox2.Controls.Add(this.label16);
            this.groupBox2.Controls.Add(this.PictureBox_InputImage);
            this.groupBox2.Controls.Add(this.Button_ImageToImage);
            this.groupBox2.Controls.Add(this.Button_RandomSeed);
            this.groupBox2.Controls.Add(this.NumericUpDown_Seed);
            this.groupBox2.Controls.Add(this.label15);
            this.groupBox2.Controls.Add(this.NumericUpDown_SampleSteps);
            this.groupBox2.Controls.Add(this.label10);
            this.groupBox2.Controls.Add(this.label9);
            this.groupBox2.Controls.Add(this.label8);
            this.groupBox2.Controls.Add(this.NumericUpDown_Height);
            this.groupBox2.Controls.Add(this.NumericUpDown_Width);
            this.groupBox2.Controls.Add(this.NumericUpDown_CFG);
            this.groupBox2.Controls.Add(this.label7);
            this.groupBox2.Controls.Add(this.label6);
            this.groupBox2.Controls.Add(this.ComboBox_SampleMethod);
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.Button_TextToImage);
            this.groupBox2.Controls.Add(this.TextBox_Prompt);
            this.groupBox2.Controls.Add(this.label2);
            this.groupBox2.Controls.Add(this.TextBox_NegativePrompt);
            this.groupBox2.Location = new System.Drawing.Point(7, 6);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(363, 399);
            this.groupBox2.TabIndex = 19;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Parameters";
            // 
            // NumericUpDown_ReDrawStrength
            // 
            this.NumericUpDown_ReDrawStrength.DecimalPlaces = 2;
            this.NumericUpDown_ReDrawStrength.Increment = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.NumericUpDown_ReDrawStrength.Location = new System.Drawing.Point(288, 335);
            this.NumericUpDown_ReDrawStrength.Maximum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.NumericUpDown_ReDrawStrength.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.NumericUpDown_ReDrawStrength.Name = "NumericUpDown_ReDrawStrength";
            this.NumericUpDown_ReDrawStrength.Size = new System.Drawing.Size(64, 21);
            this.NumericUpDown_ReDrawStrength.TabIndex = 37;
            this.NumericUpDown_ReDrawStrength.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.NumericUpDown_ReDrawStrength.Value = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(211, 344);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(41, 12);
            this.label17.TabIndex = 36;
            this.label17.Text = "Redraw";
            // 
            // NumericUpDown_ClipSkip
            // 
            this.NumericUpDown_ClipSkip.Location = new System.Drawing.Point(288, 231);
            this.NumericUpDown_ClipSkip.Maximum = new decimal(new int[] {
            10,
            0,
            0,
            0});
            this.NumericUpDown_ClipSkip.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            -2147483648});
            this.NumericUpDown_ClipSkip.Name = "NumericUpDown_ClipSkip";
            this.NumericUpDown_ClipSkip.Size = new System.Drawing.Size(66, 21);
            this.NumericUpDown_ClipSkip.TabIndex = 35;
            this.NumericUpDown_ClipSkip.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.NumericUpDown_ClipSkip.Value = new decimal(new int[] {
            1,
            0,
            0,
            -2147483648});
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(224, 234);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(59, 12);
            this.label16.TabIndex = 34;
            this.label16.Text = "Clip Skip";
            // 
            // PictureBox_InputImage
            // 
            this.PictureBox_InputImage.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.PictureBox_InputImage.Location = new System.Drawing.Point(7, 259);
            this.PictureBox_InputImage.Name = "PictureBox_InputImage";
            this.PictureBox_InputImage.Size = new System.Drawing.Size(194, 127);
            this.PictureBox_InputImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.PictureBox_InputImage.TabIndex = 33;
            this.PictureBox_InputImage.TabStop = false;
            this.PictureBox_InputImage.Click += new System.EventHandler(this.PictureBox_InputImage_Click);
            this.PictureBox_InputImage.DragDrop += new System.Windows.Forms.DragEventHandler(this.PictureBox_InputImage_DragDrop);
            this.PictureBox_InputImage.DragEnter += new System.Windows.Forms.DragEventHandler(this.PictureBox_InputImage_DragEnter);
            // 
            // Button_ImageToImage
            // 
            this.Button_ImageToImage.Location = new System.Drawing.Point(213, 363);
            this.Button_ImageToImage.Name = "Button_ImageToImage";
            this.Button_ImageToImage.Size = new System.Drawing.Size(139, 23);
            this.Button_ImageToImage.TabIndex = 32;
            this.Button_ImageToImage.Text = "Image To Image";
            this.Button_ImageToImage.UseVisualStyleBackColor = true;
            this.Button_ImageToImage.Click += new System.EventHandler(this.Button_ImageToImage_Click);
            // 
            // Button_RandomSeed
            // 
            this.Button_RandomSeed.Location = new System.Drawing.Point(145, 229);
            this.Button_RandomSeed.Name = "Button_RandomSeed";
            this.Button_RandomSeed.Size = new System.Drawing.Size(68, 23);
            this.Button_RandomSeed.TabIndex = 31;
            this.Button_RandomSeed.Text = "Random";
            this.Button_RandomSeed.UseVisualStyleBackColor = true;
            this.Button_RandomSeed.Click += new System.EventHandler(this.Button_RandomSeed_Click);
            // 
            // NumericUpDown_Seed
            // 
            this.NumericUpDown_Seed.Location = new System.Drawing.Point(57, 232);
            this.NumericUpDown_Seed.Maximum = new decimal(new int[] {
            276447231,
            23283,
            0,
            0});
            this.NumericUpDown_Seed.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            -2147483648});
            this.NumericUpDown_Seed.Name = "NumericUpDown_Seed";
            this.NumericUpDown_Seed.Size = new System.Drawing.Size(82, 21);
            this.NumericUpDown_Seed.TabIndex = 30;
            this.NumericUpDown_Seed.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.NumericUpDown_Seed.Value = new decimal(new int[] {
            1,
            0,
            0,
            -2147483648});
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(22, 237);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(29, 12);
            this.label15.TabIndex = 29;
            this.label15.Text = "Seed";
            // 
            // NumericUpDown_SampleSteps
            // 
            this.NumericUpDown_SampleSteps.Location = new System.Drawing.Point(288, 192);
            this.NumericUpDown_SampleSteps.Maximum = new decimal(new int[] {
            150,
            0,
            0,
            0});
            this.NumericUpDown_SampleSteps.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.NumericUpDown_SampleSteps.Name = "NumericUpDown_SampleSteps";
            this.NumericUpDown_SampleSteps.Size = new System.Drawing.Size(68, 21);
            this.NumericUpDown_SampleSteps.TabIndex = 28;
            this.NumericUpDown_SampleSteps.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.NumericUpDown_SampleSteps.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(243, 198);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(35, 12);
            this.label10.TabIndex = 27;
            this.label10.Text = "Steps";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(135, 201);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(11, 12);
            this.label9.TabIndex = 26;
            this.label9.Text = "H";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(22, 201);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(11, 12);
            this.label8.TabIndex = 25;
            this.label8.Text = "W";
            // 
            // NumericUpDown_Height
            // 
            this.NumericUpDown_Height.Increment = new decimal(new int[] {
            64,
            0,
            0,
            0});
            this.NumericUpDown_Height.Location = new System.Drawing.Point(158, 196);
            this.NumericUpDown_Height.Maximum = new decimal(new int[] {
            1920,
            0,
            0,
            0});
            this.NumericUpDown_Height.Minimum = new decimal(new int[] {
            256,
            0,
            0,
            0});
            this.NumericUpDown_Height.Name = "NumericUpDown_Height";
            this.NumericUpDown_Height.Size = new System.Drawing.Size(68, 21);
            this.NumericUpDown_Height.TabIndex = 24;
            this.NumericUpDown_Height.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.NumericUpDown_Height.Value = new decimal(new int[] {
            512,
            0,
            0,
            0});
            // 
            // NumericUpDown_Width
            // 
            this.NumericUpDown_Width.Increment = new decimal(new int[] {
            64,
            0,
            0,
            0});
            this.NumericUpDown_Width.Location = new System.Drawing.Point(49, 196);
            this.NumericUpDown_Width.Maximum = new decimal(new int[] {
            1920,
            0,
            0,
            0});
            this.NumericUpDown_Width.Minimum = new decimal(new int[] {
            256,
            0,
            0,
            0});
            this.NumericUpDown_Width.Name = "NumericUpDown_Width";
            this.NumericUpDown_Width.Size = new System.Drawing.Size(68, 21);
            this.NumericUpDown_Width.TabIndex = 23;
            this.NumericUpDown_Width.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.NumericUpDown_Width.Value = new decimal(new int[] {
            512,
            0,
            0,
            0});
            // 
            // NumericUpDown_CFG
            // 
            this.NumericUpDown_CFG.DecimalPlaces = 1;
            this.NumericUpDown_CFG.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.NumericUpDown_CFG.Location = new System.Drawing.Point(288, 160);
            this.NumericUpDown_CFG.Maximum = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.NumericUpDown_CFG.Minimum = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.NumericUpDown_CFG.Name = "NumericUpDown_CFG";
            this.NumericUpDown_CFG.Size = new System.Drawing.Size(70, 21);
            this.NumericUpDown_CFG.TabIndex = 22;
            this.NumericUpDown_CFG.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.NumericUpDown_CFG.Value = new decimal(new int[] {
            7,
            0,
            0,
            0});
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(224, 164);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(59, 12);
            this.label7.TabIndex = 21;
            this.label7.Text = "CFG Scale";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(22, 164);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(47, 12);
            this.label6.TabIndex = 20;
            this.label6.Text = "Sampler";
            // 
            // ComboBox_SampleMethod
            // 
            this.ComboBox_SampleMethod.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.ComboBox_SampleMethod.FormattingEnabled = true;
            this.ComboBox_SampleMethod.Location = new System.Drawing.Point(87, 161);
            this.ComboBox_SampleMethod.Name = "ComboBox_SampleMethod";
            this.ComboBox_SampleMethod.Size = new System.Drawing.Size(114, 20);
            this.ComboBox_SampleMethod.TabIndex = 19;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(33, 49);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(41, 12);
            this.label1.TabIndex = 2;
            this.label1.Text = "Prompt";
            // 
            // Button_TextToImage
            // 
            this.Button_TextToImage.Location = new System.Drawing.Point(213, 259);
            this.Button_TextToImage.Name = "Button_TextToImage";
            this.Button_TextToImage.Size = new System.Drawing.Size(141, 23);
            this.Button_TextToImage.TabIndex = 18;
            this.Button_TextToImage.Text = "Text To Image";
            this.Button_TextToImage.UseVisualStyleBackColor = true;
            this.Button_TextToImage.Click += new System.EventHandler(this.Button_TextToImage_Click);
            // 
            // TextBox_Prompt
            // 
            this.TextBox_Prompt.Location = new System.Drawing.Point(86, 20);
            this.TextBox_Prompt.Multiline = true;
            this.TextBox_Prompt.Name = "TextBox_Prompt";
            this.TextBox_Prompt.Size = new System.Drawing.Size(271, 80);
            this.TextBox_Prompt.TabIndex = 1;
            this.TextBox_Prompt.Text = "realistic, best quality, 4k, 8k, trees, beach, moon, stars, boat, ";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(21, 120);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(53, 12);
            this.label2.TabIndex = 3;
            this.label2.Text = "N_Prompt";
            // 
            // TextBox_NegativePrompt
            // 
            this.TextBox_NegativePrompt.Location = new System.Drawing.Point(86, 106);
            this.TextBox_NegativePrompt.Multiline = true;
            this.TextBox_NegativePrompt.Name = "TextBox_NegativePrompt";
            this.TextBox_NegativePrompt.Size = new System.Drawing.Size(271, 47);
            this.TextBox_NegativePrompt.TabIndex = 4;
            this.TextBox_NegativePrompt.Text = "2d, 3d, cartoon, paintings";
            // 
            // TabControl
            // 
            this.TabControl.Controls.Add(this.TabPage_CreateImage);
            this.TabControl.Controls.Add(this.TabPage_Upscale);
            this.TabControl.Location = new System.Drawing.Point(12, 165);
            this.TabControl.Name = "TabControl";
            this.TabControl.SelectedIndex = 0;
            this.TabControl.Size = new System.Drawing.Size(760, 439);
            this.TabControl.TabIndex = 25;
            // 
            // label19
            // 
            this.label19.Location = new System.Drawing.Point(211, 285);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(141, 37);
            this.label19.TabIndex = 38;
            this.label19.Text = "Please drag and drop an image into this picturebox";
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(784, 671);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.TabControl);
            this.Controls.Add(this.groupBox1);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MaximumSize = new System.Drawing.Size(800, 710);
            this.MinimizeBox = false;
            this.MinimumSize = new System.Drawing.Size(800, 710);
            this.Name = "MainForm";
            this.Text = "StableDiffusionSharpGUI";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.Form1_FormClosed);
            this.Load += new System.EventHandler(this.MainForm_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.TabPage_Upscale.ResumeLayout(false);
            this.TabPage_Upscale.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.PictureBox_UpscaleOutput)).EndInit();
            this.groupBox4.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.PictureBox_UpscaleInput)).EndInit();
            this.TabPage_CreateImage.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.PictureBox_OutputImage)).EndInit();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_ReDrawStrength)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_ClipSkip)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.PictureBox_InputImage)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_Seed)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_SampleSteps)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_Height)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_Width)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.NumericUpDown_CFG)).EndInit();
            this.TabControl.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox TextBox_ModelPath;
        private System.Windows.Forms.Button Button_ScanModelPath;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox TextBox_VaePath;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox TextBox_LoraModelDir;
        private System.Windows.Forms.Button Button_ScanVaePath;
        private System.Windows.Forms.Button Button_ScanLoraPath;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button Button_LoadModel;
        private System.Windows.Forms.CheckBox CheckBox_VaeTiling;
        private System.Windows.Forms.CheckBox CheckBox_CpuVae;
        private System.Windows.Forms.ProgressBar ProgressBar_Progress;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.TabPage TabPage_Upscale;
        private System.Windows.Forms.TabPage TabPage_CreateImage;
        private System.Windows.Forms.PictureBox PictureBox_OutputImage;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Button Button_ImageToImage;
        private System.Windows.Forms.Button Button_RandomSeed;
        private System.Windows.Forms.NumericUpDown NumericUpDown_Seed;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.NumericUpDown NumericUpDown_SampleSteps;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.NumericUpDown NumericUpDown_Height;
        private System.Windows.Forms.NumericUpDown NumericUpDown_Width;
        private System.Windows.Forms.NumericUpDown NumericUpDown_CFG;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.ComboBox ComboBox_SampleMethod;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button Button_TextToImage;
        private System.Windows.Forms.TextBox TextBox_Prompt;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox TextBox_NegativePrompt;
        private System.Windows.Forms.TabControl TabControl;
        private System.Windows.Forms.PictureBox PictureBox_InputImage;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.NumericUpDown NumericUpDown_ClipSkip;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.NumericUpDown NumericUpDown_ReDrawStrength;
        private System.Windows.Forms.PictureBox PictureBox_UpscaleOutput;
        private System.Windows.Forms.PictureBox PictureBox_UpscaleInput;
        private System.Windows.Forms.GroupBox groupBox5;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.Label label18;
        private System.Windows.Forms.Button Button_Upscale;
        private System.Windows.Forms.Button Button_ScanUpscaleModelPath;
        private System.Windows.Forms.TextBox TextBox_UpscaleModelPath;
        private System.Windows.Forms.Label label19;
    }
}

