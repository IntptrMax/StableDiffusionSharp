namespace StableDiffusionDemo_Winform
{
	partial class FormMain
	{
		/// <summary>
		///  Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		///  Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		///  Required method for Designer support - do not modify
		///  the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			groupBox1 = new GroupBox();
			Button_ModelLoad = new Button();
			Button_ModelScan = new Button();
			label1 = new Label();
			TextBox_ModelPath = new TextBox();
			tabControl1 = new TabControl();
			tabPage1 = new TabPage();
			groupBox2 = new GroupBox();
			Label_State = new Label();
			Button_Generate = new Button();
			label7 = new Label();
			label6 = new Label();
			label5 = new Label();
			NumericUpDown_Height = new NumericUpDown();
			NumericUpDown_CFG = new NumericUpDown();
			NumericUpDown_Step = new NumericUpDown();
			NumericUpDown_Width = new NumericUpDown();
			label4 = new Label();
			PictureBox_Output = new PictureBox();
			label3 = new Label();
			TextBox_NPrompt = new TextBox();
			TextBox_Prompt = new TextBox();
			label2 = new Label();
			tabPage2 = new TabPage();
			tabPage3 = new TabPage();
			groupBox1.SuspendLayout();
			tabControl1.SuspendLayout();
			tabPage1.SuspendLayout();
			groupBox2.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_Height).BeginInit();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_CFG).BeginInit();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_Step).BeginInit();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_Width).BeginInit();
			((System.ComponentModel.ISupportInitialize)PictureBox_Output).BeginInit();
			SuspendLayout();
			// 
			// groupBox1
			// 
			groupBox1.Controls.Add(Button_ModelLoad);
			groupBox1.Controls.Add(Button_ModelScan);
			groupBox1.Controls.Add(label1);
			groupBox1.Controls.Add(TextBox_ModelPath);
			groupBox1.Location = new Point(12, 12);
			groupBox1.Name = "groupBox1";
			groupBox1.Size = new Size(865, 178);
			groupBox1.TabIndex = 0;
			groupBox1.TabStop = false;
			groupBox1.Text = "Base";
			// 
			// Button_ModelLoad
			// 
			Button_ModelLoad.Location = new Point(708, 137);
			Button_ModelLoad.Name = "Button_ModelLoad";
			Button_ModelLoad.Size = new Size(101, 23);
			Button_ModelLoad.TabIndex = 3;
			Button_ModelLoad.Text = "Load Model";
			Button_ModelLoad.UseVisualStyleBackColor = true;
			Button_ModelLoad.Click += Button_ModelLoad_Click;
			// 
			// Button_ModelScan
			// 
			Button_ModelScan.Location = new Point(708, 40);
			Button_ModelScan.Name = "Button_ModelScan";
			Button_ModelScan.Size = new Size(101, 23);
			Button_ModelScan.TabIndex = 2;
			Button_ModelScan.Text = "Scan";
			Button_ModelScan.UseVisualStyleBackColor = true;
			Button_ModelScan.Click += Button_ModelScan_Click;
			// 
			// label1
			// 
			label1.AutoSize = true;
			label1.Location = new Point(18, 40);
			label1.Name = "label1";
			label1.Size = new Size(75, 17);
			label1.TabIndex = 1;
			label1.Text = "Model Path";
			// 
			// TextBox_ModelPath
			// 
			TextBox_ModelPath.Location = new Point(113, 34);
			TextBox_ModelPath.Name = "TextBox_ModelPath";
			TextBox_ModelPath.ReadOnly = true;
			TextBox_ModelPath.Size = new Size(564, 23);
			TextBox_ModelPath.TabIndex = 0;
			// 
			// tabControl1
			// 
			tabControl1.Controls.Add(tabPage1);
			tabControl1.Controls.Add(tabPage2);
			tabControl1.Controls.Add(tabPage3);
			tabControl1.Location = new Point(12, 196);
			tabControl1.Name = "tabControl1";
			tabControl1.SelectedIndex = 0;
			tabControl1.Size = new Size(865, 397);
			tabControl1.TabIndex = 1;
			// 
			// tabPage1
			// 
			tabPage1.Controls.Add(groupBox2);
			tabPage1.Location = new Point(4, 26);
			tabPage1.Name = "tabPage1";
			tabPage1.Padding = new Padding(3);
			tabPage1.Size = new Size(857, 367);
			tabPage1.TabIndex = 0;
			tabPage1.Text = "Text To Image";
			tabPage1.UseVisualStyleBackColor = true;
			// 
			// groupBox2
			// 
			groupBox2.Controls.Add(Label_State);
			groupBox2.Controls.Add(Button_Generate);
			groupBox2.Controls.Add(label7);
			groupBox2.Controls.Add(label6);
			groupBox2.Controls.Add(label5);
			groupBox2.Controls.Add(NumericUpDown_Height);
			groupBox2.Controls.Add(NumericUpDown_CFG);
			groupBox2.Controls.Add(NumericUpDown_Step);
			groupBox2.Controls.Add(NumericUpDown_Width);
			groupBox2.Controls.Add(label4);
			groupBox2.Controls.Add(PictureBox_Output);
			groupBox2.Controls.Add(label3);
			groupBox2.Controls.Add(TextBox_NPrompt);
			groupBox2.Controls.Add(TextBox_Prompt);
			groupBox2.Controls.Add(label2);
			groupBox2.Location = new Point(6, 6);
			groupBox2.Name = "groupBox2";
			groupBox2.Size = new Size(845, 355);
			groupBox2.TabIndex = 0;
			groupBox2.TabStop = false;
			groupBox2.Text = "Parameters";
			// 
			// Label_State
			// 
			Label_State.BorderStyle = BorderStyle.FixedSingle;
			Label_State.Location = new Point(6, 294);
			Label_State.Name = "Label_State";
			Label_State.Size = new Size(282, 58);
			Label_State.TabIndex = 15;
			Label_State.Text = "Please load a model first.";
			// 
			// Button_Generate
			// 
			Button_Generate.Enabled = false;
			Button_Generate.Location = new Point(294, 294);
			Button_Generate.Name = "Button_Generate";
			Button_Generate.Size = new Size(86, 55);
			Button_Generate.TabIndex = 14;
			Button_Generate.Text = "Generate";
			Button_Generate.UseVisualStyleBackColor = true;
			Button_Generate.Click += Button_Generate_Click;
			// 
			// label7
			// 
			label7.AutoSize = true;
			label7.Location = new Point(285, 267);
			label7.Name = "label7";
			label7.Size = new Size(31, 17);
			label7.TabIndex = 13;
			label7.Text = "CFG";
			// 
			// label6
			// 
			label6.AutoSize = true;
			label6.Location = new Point(167, 267);
			label6.Name = "label6";
			label6.Size = new Size(34, 17);
			label6.TabIndex = 12;
			label6.Text = "Step";
			// 
			// label5
			// 
			label5.AutoSize = true;
			label5.Location = new Point(89, 267);
			label5.Name = "label5";
			label5.Size = new Size(17, 17);
			label5.TabIndex = 11;
			label5.Text = "H";
			// 
			// NumericUpDown_Height
			// 
			NumericUpDown_Height.Increment = new decimal(new int[] { 64, 0, 0, 0 });
			NumericUpDown_Height.Location = new Point(112, 265);
			NumericUpDown_Height.Maximum = new decimal(new int[] { 2048, 0, 0, 0 });
			NumericUpDown_Height.Minimum = new decimal(new int[] { 64, 0, 0, 0 });
			NumericUpDown_Height.Name = "NumericUpDown_Height";
			NumericUpDown_Height.Size = new Size(49, 23);
			NumericUpDown_Height.TabIndex = 10;
			NumericUpDown_Height.Value = new decimal(new int[] { 512, 0, 0, 0 });
			// 
			// NumericUpDown_CFG
			// 
			NumericUpDown_CFG.Increment = new decimal(new int[] { 5, 0, 0, 65536 });
			NumericUpDown_CFG.Location = new Point(322, 265);
			NumericUpDown_CFG.Maximum = new decimal(new int[] { 25, 0, 0, 0 });
			NumericUpDown_CFG.Minimum = new decimal(new int[] { 5, 0, 0, 65536 });
			NumericUpDown_CFG.Name = "NumericUpDown_CFG";
			NumericUpDown_CFG.Size = new Size(58, 23);
			NumericUpDown_CFG.TabIndex = 9;
			NumericUpDown_CFG.Value = new decimal(new int[] { 7, 0, 0, 0 });
			// 
			// NumericUpDown_Step
			// 
			NumericUpDown_Step.Location = new Point(207, 265);
			NumericUpDown_Step.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
			NumericUpDown_Step.Name = "NumericUpDown_Step";
			NumericUpDown_Step.Size = new Size(60, 23);
			NumericUpDown_Step.TabIndex = 8;
			NumericUpDown_Step.Value = new decimal(new int[] { 20, 0, 0, 0 });
			// 
			// NumericUpDown_Width
			// 
			NumericUpDown_Width.Increment = new decimal(new int[] { 64, 0, 0, 0 });
			NumericUpDown_Width.Location = new Point(34, 265);
			NumericUpDown_Width.Maximum = new decimal(new int[] { 2048, 0, 0, 0 });
			NumericUpDown_Width.Minimum = new decimal(new int[] { 64, 0, 0, 0 });
			NumericUpDown_Width.Name = "NumericUpDown_Width";
			NumericUpDown_Width.Size = new Size(49, 23);
			NumericUpDown_Width.TabIndex = 6;
			NumericUpDown_Width.Value = new decimal(new int[] { 512, 0, 0, 0 });
			// 
			// label4
			// 
			label4.AutoSize = true;
			label4.Location = new Point(8, 267);
			label4.Name = "label4";
			label4.Size = new Size(20, 17);
			label4.TabIndex = 5;
			label4.Text = "W";
			// 
			// PictureBox_Output
			// 
			PictureBox_Output.BorderStyle = BorderStyle.FixedSingle;
			PictureBox_Output.Location = new Point(398, 22);
			PictureBox_Output.Name = "PictureBox_Output";
			PictureBox_Output.Size = new Size(432, 327);
			PictureBox_Output.SizeMode = PictureBoxSizeMode.Zoom;
			PictureBox_Output.TabIndex = 4;
			PictureBox_Output.TabStop = false;
			// 
			// label3
			// 
			label3.AutoSize = true;
			label3.Location = new Point(8, 193);
			label3.Name = "label3";
			label3.Size = new Size(66, 17);
			label3.TabIndex = 3;
			label3.Text = "N_Prompt";
			// 
			// TextBox_NPrompt
			// 
			TextBox_NPrompt.Location = new Point(78, 158);
			TextBox_NPrompt.Multiline = true;
			TextBox_NPrompt.Name = "TextBox_NPrompt";
			TextBox_NPrompt.Size = new Size(302, 87);
			TextBox_NPrompt.TabIndex = 2;
			TextBox_NPrompt.Text = "2d, 3d, cartoon, paintings";
			// 
			// TextBox_Prompt
			// 
			TextBox_Prompt.Location = new Point(78, 36);
			TextBox_Prompt.Multiline = true;
			TextBox_Prompt.Name = "TextBox_Prompt";
			TextBox_Prompt.Size = new Size(302, 104);
			TextBox_Prompt.TabIndex = 1;
			TextBox_Prompt.Text = "realistic, best quality, 4k, 8k, trees, beach, moon, stars, boat, ";
			// 
			// label2
			// 
			label2.AutoSize = true;
			label2.Location = new Point(8, 90);
			label2.Name = "label2";
			label2.Size = new Size(51, 17);
			label2.TabIndex = 0;
			label2.Text = "Prompt";
			// 
			// tabPage2
			// 
			tabPage2.Location = new Point(4, 26);
			tabPage2.Name = "tabPage2";
			tabPage2.Padding = new Padding(3);
			tabPage2.Size = new Size(857, 367);
			tabPage2.TabIndex = 1;
			tabPage2.Text = "Image To Image";
			tabPage2.UseVisualStyleBackColor = true;
			// 
			// tabPage3
			// 
			tabPage3.Location = new Point(4, 26);
			tabPage3.Name = "tabPage3";
			tabPage3.Padding = new Padding(3);
			tabPage3.Size = new Size(857, 367);
			tabPage3.TabIndex = 2;
			tabPage3.Text = "Restore";
			tabPage3.UseVisualStyleBackColor = true;
			// 
			// FormMain
			// 
			AutoScaleDimensions = new SizeF(7F, 17F);
			AutoScaleMode = AutoScaleMode.Font;
			ClientSize = new Size(889, 605);
			Controls.Add(tabControl1);
			Controls.Add(groupBox1);
			Name = "FormMain";
			Text = "Stabel SDUnet Sharp";
			Load += FormMain_Load;
			groupBox1.ResumeLayout(false);
			groupBox1.PerformLayout();
			tabControl1.ResumeLayout(false);
			tabPage1.ResumeLayout(false);
			groupBox2.ResumeLayout(false);
			groupBox2.PerformLayout();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_Height).EndInit();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_CFG).EndInit();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_Step).EndInit();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_Width).EndInit();
			((System.ComponentModel.ISupportInitialize)PictureBox_Output).EndInit();
			ResumeLayout(false);
		}

		#endregion

		private GroupBox groupBox1;
		private Button Button_ModelScan;
		private Label label1;
		private TextBox TextBox_ModelPath;
		private TabControl tabControl1;
		private TabPage tabPage1;
		private Button Button_ModelLoad;
		private GroupBox groupBox2;
		private PictureBox PictureBox_Output;
		private Label label3;
		private TextBox TextBox_NPrompt;
		private TextBox TextBox_Prompt;
		private Label label2;
		private NumericUpDown NumericUpDown_Width;
		private Label label4;
		private Button Button_Generate;
		private Label label7;
		private Label label6;
		private Label label5;
		private NumericUpDown NumericUpDown_Height;
		private NumericUpDown NumericUpDown_CFG;
		private NumericUpDown NumericUpDown_Step;
		private Label Label_State;
		private TabPage tabPage2;
		private TabPage tabPage3;
	}
}
